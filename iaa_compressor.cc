// Copyright (C) 2022 Intel Corporation

// SPDX-License-Identifier: Apache-2.0

#include "iaa_compressor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <x86intrin.h>
#include <numa.h>
#include <numaif.h>

#include "logging/logging.h"
#include "qpl/qpl.h"
#include "qpl/c_api/cxl.h"
#include <dlfcn.h>
#include "rocksdb/compressor.h"
#include "rocksdb/memory_allocator.h"
#include "rocksdb/configurable.h"
#include "rocksdb/env.h"
#include "rocksdb/utilities/options_type.h"
#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {

typedef uint8_t* (*cxl_get_start_fn)();
typedef size_t (*cxl_get_size_fn)();
typedef void* (*cxl_alloc_fn)(size_t);

static uint8_t* g_cxl_pool_start = nullptr;
static size_t g_cxl_pool_size = 0;
static cxl_alloc_fn g_cxl_alloc = nullptr;

struct CxlConfig {
  bool enabled = false;
  std::string mode = "cxl";
  int numa_node = 0;
  std::string server_ip = "127.0.0.1";
  std::string cxl_bdf = "0000:40:00.1";
  int32_t cxl_numa_id = -102;
  qpl_path_t execution_path = qpl_path_pool;
};

static CxlConfig& GetCxlConfig() {
  static CxlConfig config;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    const char* cxl_enabled_env = std::getenv("QPL_CXL_ENABLED");
    if (cxl_enabled_env && (std::string(cxl_enabled_env) == "1" ||
                            std::string(cxl_enabled_env) == "true")) {
      config.enabled = true;

      const char* mode_env = std::getenv("QPL_CXL_MODE");
      if (mode_env) config.mode = mode_env;

      const char* numa_env = std::getenv("QPL_CXL_NUMA_NODE");
      if (numa_env) config.numa_node = std::atoi(numa_env);

      const char* ip_env = std::getenv("QPL_CXL_SERVER_IP");
      if (ip_env) config.server_ip = ip_env;

      const char* bdf_env = std::getenv("QPL_CXL_BDF");
      if (bdf_env) config.cxl_bdf = bdf_env;

      // Resolve CXL NUMA ID
      if (config.mode == "local") config.cxl_numa_id = -106;
      else if (config.mode == "local_umwait") config.cxl_numa_id = -107;
      else if (config.mode == "cxl") config.cxl_numa_id = -102;
      else if (config.mode == "cxl_umwait") config.cxl_numa_id = -103;
      else if (config.mode == "cpu") config.cxl_numa_id = -104;
      else if (config.mode == "rdma") config.cxl_numa_id = -105;
      else if (config.mode == "combined") config.cxl_numa_id = -108;
      else if (config.mode == "combined_umwait") config.cxl_numa_id = -109;

      // Resolve execution path
      config.execution_path = (config.cxl_numa_id <= -102 && config.cxl_numa_id >= -109)
                                  ? qpl_path_pool
                                  : qpl_path_hardware;

      // Initialize CXL Proxy client
      std::string final_ip = config.server_ip;
      if (config.mode == "combined" || config.mode == "combined_umwait") {
        final_ip = "combined:" + final_ip;
      }
      qpl_cxl_initialize(final_ip.c_str(), config.cxl_bdf.c_str(), config.numa_node);

      // Register the global 1GB mempool now that CXL client is initialized
      uint8_t* pool_start = nullptr;
      size_t pool_size = 0;

      cxl_get_start_fn get_start = (cxl_get_start_fn)dlsym(RTLD_DEFAULT, "cxl_mempool_get_start");
      cxl_get_size_fn get_size = (cxl_get_size_fn)dlsym(RTLD_DEFAULT, "cxl_mempool_get_size");

      if (get_start && get_size) {
        pool_start = get_start();
        pool_size = get_size();
        g_cxl_pool_start = pool_start;
        g_cxl_pool_size = pool_size;
      }
      
      g_cxl_alloc = (cxl_alloc_fn)dlsym(RTLD_DEFAULT, "cxl_mempool_alloc");

      if (pool_start && pool_size > 0) {
        uint64_t iova;
        qpl_status reg_status = qpl_cxl_register_buffer(pool_start, pool_size, &iova);
        if (reg_status != QPL_STS_OK) {
          std::cerr << "[IaaCompressor] WARNING: Failed to pre-register the 1GB mempool! Fast-path will fail." << std::endl;
        } else {
          std::cerr << "[IaaCompressor] Successfully pre-registered CXL Mempool (size " << pool_size << ") at IOVA " << std::hex << iova << std::dec << std::endl;
          std::atexit([]() {
            cxl_get_start_fn get_start_atexit = (cxl_get_start_fn)dlsym(RTLD_DEFAULT, "cxl_mempool_get_start");
            if (get_start_atexit) {
              uint8_t* ptr = get_start_atexit();
              if (ptr) qpl_cxl_deregister_buffer(ptr);
            }
          });
        }
      } else {
        std::cerr << "[IaaCompressor] WARNING: CXL Mempool not detected via LD_PRELOAD. Fast-path registrations will bypass the pool." << std::endl;
      }
    }
  });
  return config;
}

// Error messages
#define MEMORY_ALLOCATION_ERROR "memory allocation error"
#define JOB_INIT_ERROR "job init error"
#define QPL_STATUS(status) "QPL status " + std::to_string(status)

extern "C" FactoryFunc<Compressor> iaa_compressor_reg;

FactoryFunc<Compressor> iaa_compressor_reg =
    ObjectLibrary::Default()->AddFactory<Compressor>(
        "com.intel.iaa_compressor_rocksdb",
        [](const std::string& /* uri */,
           std::unique_ptr<Compressor>* compressor, std::string* /* errmsg */) {
          *compressor = NewIAACompressor();
          return compressor->get();
        });

std::unordered_map<std::string, qpl_path_t> execution_paths{
    {"auto", qpl_path_auto},
    {"hw", qpl_path_hardware},
    {"sw", qpl_path_software}};

enum qpl_compression_mode { dynamic_mode, fixed_mode };

std::unordered_map<std::string, qpl_compression_mode> compression_modes{
    {"dynamic", dynamic_mode}, {"fixed", fixed_mode}};

struct IAACompressorOptions {
  static const char* kName() { return "IAACompressorOptions"; };
  qpl_path_t execution_path = qpl_path_auto;
  qpl_compression_mode compression_mode = dynamic_mode;
  bool verify = false;
  int level = 0;
  uint32_t parallel_threads = 1;
};

static std::unordered_map<std::string, OptionTypeInfo>
    iaa_compressor_type_info = {
        {"execution_path",
         OptionTypeInfo::Enum(
             offsetof(struct IAACompressorOptions, execution_path),
             &execution_paths)},
        {"compression_mode",
         OptionTypeInfo::Enum(
             offsetof(struct IAACompressorOptions, compression_mode),
             &compression_modes)},
        {"verify",
         {offsetof(struct IAACompressorOptions, verify), OptionType::kBoolean,
          OptionVerificationType::kNormal, OptionTypeFlags::kNone}},
        {"level",
         {offsetof(struct IAACompressorOptions, level), OptionType::kInt,
          OptionVerificationType::kNormal, OptionTypeFlags::kNone}},
        {"parallel_threads",
         {offsetof(struct IAACompressorOptions, parallel_threads),
          OptionType::kUInt32T, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}}};

class IAAJob {
 public:
  IAAJob() : jobs_(4, nullptr), sizes_(4, 0) {
    InitJob(qpl_path_hardware);
    InitJob(qpl_path_software);
    InitJob(qpl_path_auto);
    InitJob(qpl_path_pool);
  }

  ~IAAJob() {
    auto& config = GetCxlConfig();
    for (size_t i = 0; i < jobs_.size(); ++i) {
      qpl_job* job = jobs_[i];
      if (job != nullptr) {
        qpl_fini_job(job);
        if (config.enabled && i == qpl_path_pool) {
          qpl_cxl_deregister_buffer(job);
          numa_free(job, sizes_[i]);
        } else {
          delete[] reinterpret_cast<char*>(job);
        }
      }
    }
  }

  qpl_job* GetJob(qpl_path_t execution_path) { return jobs_[execution_path]; }

 private:
  void InitJob(qpl_path_t execution_path) {
    uint32_t size;
    qpl_status status = qpl_get_job_size(execution_path, &size);
    if (status != QPL_STS_OK) {
      std::cerr << "[IaaCompressor] InitJob: qpl_get_job_size failed for path " << execution_path << " with status " << status << std::endl;
      jobs_[execution_path] = nullptr;
      return;
    }
    sizes_[execution_path] = size;
    try {
      auto& config = GetCxlConfig();
      if (config.enabled && execution_path == qpl_path_pool) {
        void* ptr = numa_alloc_onnode(size, config.numa_node);
        if (!ptr) {
          jobs_[execution_path] = nullptr;
          return;
        }
        std::memset(ptr, 0, size);
        uint64_t iova;
        qpl_status reg_status = qpl_cxl_register_buffer(ptr, size, &iova);
        if (reg_status != QPL_STS_OK) {
          std::cerr << "[IaaCompressor] InitJob: qpl_cxl_register_buffer failed for job buffer with status " << reg_status << std::endl;
          numa_free(ptr, size);
          jobs_[execution_path] = nullptr;
          return;
        }
        jobs_[execution_path] = reinterpret_cast<qpl_job*>(ptr);
      } else {
        jobs_[execution_path] = reinterpret_cast<qpl_job*>(new char[size]);
      }
    } catch (...) {
      jobs_[execution_path] = nullptr;
      return;
    }
    status = qpl_init_job(execution_path, jobs_[execution_path]);
    if (status != QPL_STS_OK) {
      std::cerr << "[IaaCompressor] InitJob: qpl_init_job failed for path " << execution_path << " with status " << status << std::endl;
      if (GetCxlConfig().enabled && execution_path == qpl_path_pool) {
        qpl_cxl_deregister_buffer(jobs_[execution_path]);
        numa_free(jobs_[execution_path], size);
      } else {
        delete[] reinterpret_cast<char*>(jobs_[execution_path]);
      }
      jobs_[execution_path] = nullptr;
    } else {
      if (GetCxlConfig().enabled && execution_path == qpl_path_pool) {
        jobs_[execution_path]->numa_id = GetCxlConfig().cxl_numa_id;
      }
    }
  }

  std::vector<qpl_job*> jobs_;
  std::vector<uint32_t> sizes_;
};

class IAACompressor : public Compressor {
 public:
  IAACompressor() {
    RegisterOptions(&options_, &iaa_compressor_type_info);

#ifndef NDEBUG
    Status s =
        Env::Default()->NewLogger("/tmp/iaa_compressor_log.txt", &logger_);
    if (s.ok()) {
      logger_->SetInfoLogLevel(DEBUG_LEVEL);
    }
#endif
  };

  static const char* kClassName() { return "com.intel.iaa_compressor_rocksdb"; }

  const char* Name() const override { return kClassName(); }

  bool DictCompressionSupported() const override { return false; }

  uint32_t GetParallelThreads() const override {
    return options_.parallel_threads;
  };

  Status Compress(const CompressionInfo& /* info */, const Slice& input,
                  std::string* output) override {
    // Max size of a RocksDB block is 4GiB
    uint32_t output_header_length = EncodeSize(input.size(), output);

    // If data is incompressible, QPL returns stored blocks
    // A stored block is at most 2^16-1 bytes in size and it has a 5-byte header
    // So, in the worst case, data grows by 5*ceil(input.size()/65535)
    size_t input_length = input.size();
    size_t output_length =
        output_header_length + input_length +
        (input_length / 65535 + (input_length % 65535 != 0)) * 5;
    if (output_length > std::numeric_limits<uint32_t>::max()) {
      // Attempt compression with largest possible buffer. QPL will return an
      // error if not sufficient.
      output_length = std::numeric_limits<uint32_t>::max();
    }
    output->resize(output_length);

    qpl_status status;
    qpl_compression_levels level = GetQplLevel(options_.level);
    qpl_path_t execution_path = options_.execution_path;

    if (level == qpl_high_level && execution_path == qpl_path_hardware) {
      execution_path = qpl_path_software;
    }

    auto& config = GetCxlConfig();
    if (execution_path != qpl_path_software && config.enabled) {
      execution_path = config.execution_path;
    }

    qpl_job* job = job_.GetJob(execution_path);
    if (job == nullptr) {
      std::cerr << "[IaaCompressor] Compress: job is nullptr for path " << execution_path << std::endl;
      return Status::Corruption(JOB_INIT_ERROR);
    }

    uint8_t* source =
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input.data()));
    uint8_t* destination =
        reinterpret_cast<uint8_t*>(&(*output)[0] + output_header_length);

    bool src_in_pool = g_cxl_pool_start && source >= g_cxl_pool_start && source < g_cxl_pool_start + g_cxl_pool_size;
    bool dst_in_pool = g_cxl_pool_start && destination >= g_cxl_pool_start && destination < g_cxl_pool_start + g_cxl_pool_size;
    
    thread_local uint8_t* bounce_src = nullptr;
    thread_local uint8_t* bounce_dst = nullptr;

    if (!src_in_pool && config.enabled && execution_path == qpl_path_pool) {
      if (input.size() > 4096) {
        std::cerr << "[QPL CXL] WARNING: Compress source buffer bypasses pool and is > 4KB (" << input.size() << " bytes)!" << std::endl;
      }
      if (input.size() <= 8192) {
        if (!bounce_src && g_cxl_alloc) bounce_src = (uint8_t*)g_cxl_alloc(8192);
        if (bounce_src) {
          std::memcpy(bounce_src, source, input.size());
          source = bounce_src;
        }
      }
    }

    size_t avail_out = output_length - output_header_length;
    if (!dst_in_pool && config.enabled && execution_path == qpl_path_pool) {
      if (avail_out > 4096) {
        std::cerr << "[QPL CXL] WARNING: Compress destination buffer bypasses pool and is > 4KB (" << avail_out << " bytes)!" << std::endl;
      }
      if (avail_out <= 8192) {
        if (!bounce_dst && g_cxl_alloc) bounce_dst = (uint8_t*)g_cxl_alloc(8192);
        if (bounce_dst) {
          destination = bounce_dst;
          avail_out = 8192; // Give hardware the full buffer size to avoid QPL_STS_MORE_OUTPUT_NEEDED
        }
      }
    }

    job->next_in_ptr = source;
    job->available_in = input.size();
    job->next_out_ptr = destination;
    job->available_out = avail_out;
    job->level = level;
    job->op = qpl_op_compress;
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;
    if (!options_.verify) {
      job->flags |= QPL_FLAG_OMIT_VERIFY;
    }
    job->huffman_table = nullptr;
    job->dictionary = nullptr;

    if (options_.compression_mode == dynamic_mode) {
      job->flags |= QPL_FLAG_DYNAMIC_HUFFMAN;
    }

    // bool use_cxl = config.enabled && (execution_path == qpl_path_pool || execution_path == qpl_path_hardware);
    // if (use_cxl) {
    //   uint64_t src_iova, dst_iova;
    //   qpl_status r1 = qpl_cxl_register_buffer(source, input.size(), &src_iova);
    //   qpl_status r2 = qpl_cxl_register_buffer(destination, output_length - output_header_length, &dst_iova);
    //   std::printf("[iaa_compressor] Compress registration: src=%p (size=%zu, status=%d), dst=%p (size=%zu, status=%d)\n",
    //               (void*)source, input.size(), (int)r1, (void*)destination, output_length - output_header_length, (int)r2);
    // }

    status = QPL_STS_QUEUES_ARE_BUSY_ERR;
    while (status == QPL_STS_QUEUES_ARE_BUSY_ERR) {
      status = qpl_submit_job(job);
    }
    if (status == QPL_STS_OK) {
      status = qpl_wait_job(job);
    }

    if (status != QPL_STS_OK) {
      std::cerr << "[IaaCompressor] Compress: qpl_execute_job failed with status " << status << std::endl;
      return Status::Corruption(QPL_STATUS(status));
    }

    // After remote CXL IAA completes, the compressed output lives in physical
    // CXL memory but the CPU cache may hold stale data from a previous
    // compression that reused the same bounce_dst. Invalidate before reading.
    if (config.enabled && execution_path == qpl_path_pool) {
      for (uint32_t i = 0; i < job->total_out; i += 64) {
        _mm_clflushopt(destination + i);
      }
      _mm_mfence();
    }

    if (!dst_in_pool && destination == bounce_dst && config.enabled && execution_path == qpl_path_pool) {
      if (output_header_length + job->total_out > output->size()) {
        output->resize(output_header_length + job->total_out);
      }
      std::memcpy(reinterpret_cast<uint8_t*>(&(*output)[0] + output_header_length), bounce_dst, job->total_out);
    }

    output->resize(output_header_length + job->total_out);
    Debug(logger_, "Compress - input size: %lu - output size: %u\n",
          input.size(), job->total_out);

    return Status::OK();
  }

  Status Uncompress(const UncompressionInfo& info, const char* input,
                    size_t input_length, char** output,
                    size_t* output_length) override {
    // Extract uncompressed size
    uint32_t encoded_output_length = 0;
    if (!DecodeSize(&input, &input_length, &encoded_output_length)) {
      return Status::Corruption("size decoding error");
    }

    // Memory allocator may return null pointer or throw bad_alloc exception
    try {
      *output = Allocate(encoded_output_length, info.GetMemoryAllocator());
      if (*output == nullptr) {
        return Status::Corruption(MEMORY_ALLOCATION_ERROR);
      }
    } catch (std::bad_alloc& e) {
      return Status::Corruption(MEMORY_ALLOCATION_ERROR);
    }

    qpl_status status;
    qpl_path_t execution_path = options_.execution_path;
    auto& config = GetCxlConfig();
    if (execution_path != qpl_path_software && config.enabled) {
      execution_path = config.execution_path;
    }

    qpl_job* job = job_.GetJob(execution_path);
    if (job == nullptr) {
      return Status::Corruption(JOB_INIT_ERROR);
    }

    uint8_t* source =
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input));
    uint8_t* destination = reinterpret_cast<uint8_t*>(*output);

    bool src_in_pool = g_cxl_pool_start && source >= g_cxl_pool_start && source < g_cxl_pool_start + g_cxl_pool_size;
    bool dst_in_pool = g_cxl_pool_start && destination >= g_cxl_pool_start && destination < g_cxl_pool_start + g_cxl_pool_size;
    
    thread_local uint8_t* bounce_src = nullptr;
    thread_local uint8_t* bounce_dst = nullptr;

    if (!src_in_pool && config.enabled && execution_path == qpl_path_pool) {
      if (input_length > 4096) {
        std::cerr << "[QPL CXL] WARNING: Uncompress source buffer bypasses pool and is > 4KB (" << input_length << " bytes)!" << std::endl;
      }
      if (input_length <= 8192) {
        if (!bounce_src && g_cxl_alloc) bounce_src = (uint8_t*)g_cxl_alloc(8192);
        if (bounce_src) {
          std::memcpy(bounce_src, source, input_length);
          source = bounce_src;
        }
      }
    }

    if (!dst_in_pool && config.enabled && execution_path == qpl_path_pool) {
      if (encoded_output_length > 4096) {
        std::cerr << "[QPL CXL] WARNING: Uncompress destination buffer bypasses pool and is > 4KB (" << encoded_output_length << " bytes)!" << std::endl;
      }
      if (encoded_output_length <= 8192) {
        if (!bounce_dst && g_cxl_alloc) bounce_dst = (uint8_t*)g_cxl_alloc(8192);
        if (bounce_dst) {
          destination = bounce_dst;
        }
      }
    }

    job->next_in_ptr = source;
    job->available_in = input_length;
    job->next_out_ptr = destination;
    job->available_out = encoded_output_length;
    job->op = qpl_op_decompress;
    job->huffman_table = nullptr;
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;


    status = QPL_STS_QUEUES_ARE_BUSY_ERR;
    while (status == QPL_STS_QUEUES_ARE_BUSY_ERR) {
      status = qpl_submit_job(job);
    }
    if (status == QPL_STS_OK) {
      status = qpl_wait_job(job);
    }



    if (status != QPL_STS_OK) {
      return Status::Corruption(QPL_STATUS(status));
    } else if (job->total_out != encoded_output_length) {
      return Status::Corruption("size mismatch");
    }

    // Invalidate CPU cache for destination after remote CXL IAA writes to it
    if (config.enabled && execution_path == qpl_path_pool) {
      for (uint32_t i = 0; i < job->total_out; i += 64) {
        _mm_clflushopt(destination + i);
      }
      _mm_mfence();
    }

    if (!dst_in_pool && destination == bounce_dst && config.enabled && execution_path == qpl_path_pool) {
      std::memcpy(reinterpret_cast<uint8_t*>(*output), bounce_dst, job->total_out);
    }

    *output_length = job->total_out;
    Debug(logger_, "Uncompress - input size: %lu - output size: %u\n",
          input_length, job->total_out);

    return Status::OK();
  }

  bool IsDictEnabled() const override { return false; }

 private:
  IAACompressorOptions options_;
  static thread_local IAAJob job_;
  std::shared_ptr<Logger> logger_;

  uint32_t EncodeSize(size_t length, std::string* output) {
    PutVarint32(output, length);
    return output->size();
  }

  bool DecodeSize(const char** input, size_t* input_length,
                  uint32_t* output_length) {
    auto new_input =
        GetVarint32Ptr(*input, *input + *input_length, output_length);
    if (new_input == nullptr) {
      return false;
    }
    *input_length -= (new_input - *input);
    *input = new_input;
    return true;
  }

  int GetLevel() const override { return options_.level; }

  qpl_compression_levels GetQplLevel(int level) {
    if (level == 0 || level == CompressionOptions::kDefaultCompressionLevel) {
      return qpl_default_level;
    } else {
      return qpl_high_level;
    }
  }
};

// Reuse job structs across calls. Have one struct per thread and execution path
// (hw, sw, auto).
thread_local IAAJob IAACompressor::job_;

std::unique_ptr<Compressor> NewIAACompressor() {
  return std::unique_ptr<Compressor>(new IAACompressor());
}

}  // namespace ROCKSDB_NAMESPACE
