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

extern "C" {
void* cxl_mempool_alloc(size_t size);
void cxl_mempool_free(void* ptr);
uint8_t* cxl_mempool_get_start(void);
size_t cxl_mempool_get_size(void);
size_t cxl_mempool_get_sgl_count(void);
struct CxlSglEntryLocal { void* va; size_t size; };
void cxl_mempool_get_sgl(CxlSglEntryLocal* out_entries);
}

namespace ROCKSDB_NAMESPACE {

static uint8_t* g_cxl_pool_start = nullptr;
static size_t g_cxl_pool_size = 0;

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

      g_cxl_pool_start = cxl_mempool_get_start();
      g_cxl_pool_size = cxl_mempool_get_size();
      std::cerr << "[IaaCompressor] SGL functions detected! pool_start=" << (void*)g_cxl_pool_start << " pool_size=" << g_cxl_pool_size << std::endl;

      size_t count = cxl_mempool_get_sgl_count();
      std::vector<CxlSglEntryLocal> entries(count);
      cxl_mempool_get_sgl(entries.data());
      for (size_t i = 0; i < count; ++i) {
        uint64_t iova = 0;
        qpl_status reg_st = qpl_cxl_register_buffer(reinterpret_cast<uint8_t*>(entries[i].va), entries[i].size, &iova);
        std::cerr << "[IaaCompressor] Segment " << i << " va=" << entries[i].va 
                  << " size=" << entries[i].size << " reg_status=" << reg_st 
                  << " iova=0x" << std::hex << iova << std::dec << std::endl;
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

class CxlMemoryAllocator : public MemoryAllocator {
 public:
  static const char* kClassName() { return "CxlMemoryAllocator"; }
  const char* Name() const override { return kClassName(); }

  void* Allocate(size_t size) override {
    return cxl_mempool_alloc(size);
  }

  void Deallocate(void* p) override {
    cxl_mempool_free(p);
  }
};

extern "C" FactoryFunc<MemoryAllocator> iaa_memory_allocator_reg;

FactoryFunc<MemoryAllocator> iaa_memory_allocator_reg =
    ObjectLibrary::Default()->AddFactory<MemoryAllocator>(
        "iaa_memory_allocator",
        [](const std::string& /* uri */,
           std::unique_ptr<MemoryAllocator>* allocator, std::string* /* errmsg */) {
          allocator->reset(new CxlMemoryAllocator());
          return allocator->get();
        });

std::unordered_map<std::string, qpl_path_t> execution_paths{
    {"auto", qpl_path_auto},
    {"hw", qpl_path_hardware},
    {"sw", qpl_path_software}};

enum qpl_compression_mode { dynamic_mode, fixed_mode, canned_mode };

std::unordered_map<std::string, qpl_compression_mode> compression_modes{
    {"dynamic", dynamic_mode}, {"fixed", fixed_mode}, {"canned", canned_mode}};

static qpl_huffman_table_t GetGlobalCannedHuffmanTable() {
  static qpl_huffman_table_t huffman_table = nullptr;
  static std::once_flag flag;
  std::call_once(flag, []() {
    qpl_status status = qpl_deflate_huffman_table_create(combined_table_type, qpl_path_auto, DEFAULT_ALLOCATOR_C, &huffman_table);
    if (status != QPL_STS_OK) {
      std::cerr << "[IaaCompressor] Error creating canned Huffman table: " << status << std::endl;
      return;
    }
    qpl_histogram histogram;
    std::memset(&histogram, 0, sizeof(histogram));
    for (int i = 0; i < 286; ++i) histogram.literal_lengths[i] = 1;
    for (int i = 0; i < 30; ++i) histogram.distances[i] = 1;
    
    status = qpl_huffman_table_init_with_histogram(huffman_table, &histogram);
    if (status != QPL_STS_OK) {
      std::cerr << "[IaaCompressor] Error initializing canned Huffman table: " << status << std::endl;
    } else {
      std::cout << "[IaaCompressor] Successfully initialized global canned Huffman table." << std::endl;
    }
  });
  return huffman_table;
}

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
          cxl_mempool_free(job);
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
        void* ptr = cxl_mempool_alloc(size);
        if (!ptr) {
          jobs_[execution_path] = nullptr;
          return;
        }
        std::memset(ptr, 0, size);
        uint64_t iova;
        qpl_status reg_status = qpl_cxl_register_buffer(ptr, size, &iova);
        if (reg_status != QPL_STS_OK) {
          std::cerr << "[IaaCompressor] InitJob: qpl_cxl_register_buffer failed for job buffer with status " << reg_status << std::endl;
          cxl_mempool_free(ptr);
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
      bool should_log = (execution_path == qpl_path_software) || 
                        (GetCxlConfig().enabled && execution_path == qpl_path_pool);
      if (should_log) {
        std::cerr << "[IaaCompressor] InitJob: qpl_init_job failed for path " << execution_path << " with status " << status << std::endl;
      }
      if (GetCxlConfig().enabled && execution_path == qpl_path_pool) {
        qpl_cxl_deregister_buffer(jobs_[execution_path]);
        cxl_mempool_free(jobs_[execution_path]);
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
    thread_local size_t bounce_src_cap = 0;
    thread_local uint8_t* bounce_dst = nullptr;
    thread_local size_t bounce_dst_cap = 0;

    if (!src_in_pool && config.enabled && execution_path == qpl_path_pool) {
      if (input.size() > bounce_src_cap) {
        size_t new_cap = std::max(input.size() * 2, static_cast<size_t>(65536));
        bounce_src = (uint8_t*)cxl_mempool_alloc(new_cap);
        if (bounce_src) bounce_src_cap = new_cap;
      }
      if (bounce_src) {
        std::memcpy(bounce_src, source, input.size());
        source = bounce_src;
      }
    }

    size_t avail_out = output_length - output_header_length;
    if (!dst_in_pool && config.enabled && execution_path == qpl_path_pool) {
      if (avail_out > bounce_dst_cap) {
        size_t new_cap = std::max(avail_out * 2, static_cast<size_t>(65536));
        bounce_dst = (uint8_t*)cxl_mempool_alloc(new_cap);
        if (bounce_dst) bounce_dst_cap = new_cap;
      }
      if (bounce_dst) {
        destination = bounce_dst;
        avail_out = bounce_dst_cap;
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
    } else if (options_.compression_mode == canned_mode) {
      job->flags |= QPL_FLAG_CANNED_MODE;
      job->huffman_table = GetGlobalCannedHuffmanTable();
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
    thread_local size_t bounce_src_cap = 0;
    thread_local uint8_t* bounce_dst = nullptr;
    thread_local size_t bounce_dst_cap = 0;

    if (!src_in_pool && config.enabled && execution_path == qpl_path_pool) {
      if (input_length > bounce_src_cap) {
        size_t new_cap = std::max(input_length * 2, static_cast<size_t>(65536));
        bounce_src = (uint8_t*)cxl_mempool_alloc(new_cap);
        if (bounce_src) bounce_src_cap = new_cap;
      }
      if (bounce_src) {
        std::memcpy(bounce_src, source, input_length);
        source = bounce_src;
      }
    }

    if (!dst_in_pool && config.enabled && execution_path == qpl_path_pool) {
      if (encoded_output_length > bounce_dst_cap) {
        size_t new_cap = std::max(static_cast<size_t>(encoded_output_length) * 2, static_cast<size_t>(65536));
        bounce_dst = (uint8_t*)cxl_mempool_alloc(new_cap);
        if (bounce_dst) bounce_dst_cap = new_cap;
      }
      if (bounce_dst) {
        destination = bounce_dst;
      }
    }



    job->next_in_ptr = source;
    job->available_in = input_length;
    job->next_out_ptr = destination;
    job->available_out = encoded_output_length;
    job->op = qpl_op_decompress;
    job->huffman_table = nullptr;
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;
    if (options_.compression_mode == canned_mode) {
      job->flags |= QPL_FLAG_CANNED_MODE;
      job->huffman_table = GetGlobalCannedHuffmanTable();
    }


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
