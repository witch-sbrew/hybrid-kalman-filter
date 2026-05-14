#pragma once

#include <vector>
#include <future>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include "kfcpu.hpp"
#include "gpuint.hpp"

namespace kf {

    struct KFInstance {
        int filter_count = 0;
        int step_count = 0;
        const int state_dim = 64;

        double* initial_states = nullptr;
        double* measurements = nullptr;
        double* output = nullptr;
    };

    struct ExStats {
        float cpu_ms  = 0.f;
        float gpu_ms  = 0.f;
 
        float transfer_ms = 0.f;
        float idle_ms = 0.f;
 
        int N_cpu = 0;
        int N_gpu = 0;

        int omp_threads_used = 0;
    };

class GpuContext {
public:
    explicit GpuContext(int device = 0) : device_(device) {
        cudaSetDevice(device_);
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        cudaEventCreate(&ev_start_);
        cudaEventCreate(&ev_stop_);
        cudaEventCreate(&ev_h2d_done_);
    }
 
    ~GpuContext() {
        cudaStreamSynchronize(stream_);
        cudaEventDestroy(ev_start_);
        cudaEventDestroy(ev_stop_);
        cudaEventDestroy(ev_h2d_done_);
        cudaStreamDestroy(stream_);
    }
 
    GpuContext(const GpuContext&)            = delete;
    GpuContext& operator=(const GpuContext&) = delete;
    GpuContext(GpuContext&&)                 = default;
    GpuContext& operator=(GpuContext&&)      = default;
 
    cudaStream_t stream()      const { return stream_; }
    cudaEvent_t  ev_start()    const { return ev_start_; }
    cudaEvent_t  ev_stop()     const { return ev_stop_; }
    cudaEvent_t  ev_h2d_done() const { return ev_h2d_done_; }
    int          device()      const { return device_; }
 
private:
    int          device_;
    cudaStream_t stream_     = nullptr;
    cudaEvent_t  ev_start_   = nullptr;
    cudaEvent_t  ev_stop_    = nullptr;
    cudaEvent_t  ev_h2d_done_= nullptr;
};

struct SchedulerConfig {
    int T_cpu = 16;
    int T1_time = 30; 

    int gpu_alloc_time = 400;

};

class HScheduler {
    public:

        explicit HScheduler(bool gpu_only, SchedulerConfig cfg = {}): cfg_(cfg), gpu_ctx_(0), gpu_only_(gpu_only){

        }

        ExStats run(const KFInstance& job){
            ExStats stats;
            //CHANGE
            int n_cpu = 0, n_gpu = 0;
            
            if(gpu_only_) {
                n_gpu = job.filter_count;
            } else {
            if(job.filter_count + job.step_count <= 64 + 32){
                n_cpu = job.filter_count;
                n_gpu = 0;
            }
            else if(job.step_count >= 32){
                n_cpu = 0;
                n_gpu = job.filter_count;
            } else {
                n_cpu = 32;
                n_gpu = job.filter_count - n_cpu;
            }
            }

            stats.N_cpu = n_cpu;
            stats.N_gpu = n_gpu;
            

            std::future<float> cpu_future;

            if (n_cpu > 0) {
            cpu_future = std::async(
                std::launch::async,
                &HScheduler::run_cpu_partition, this,
                std::cref(job), 0, n_cpu,
                std::ref(stats.omp_threads_used), n_cpu
            );
            }

            auto wall_start = std::chrono::high_resolution_clock::now();
            if (n_gpu > 0) {
                launch_gpu_partition(job, n_cpu, n_gpu, stats);
            }

             
 
            if (n_gpu > 0) {
                cudaStreamSynchronize(gpu_ctx_.stream());
                cudaEventElapsedTime(&stats.gpu_ms,
                                 gpu_ctx_.ev_start(),
                                 gpu_ctx_.ev_stop());
            }
 
            if (n_cpu > 0) {
                stats.cpu_ms = cpu_future.get(); 
            }
 
            auto wall_end = std::chrono::high_resolution_clock::now();
            float wall_ms = std::chrono::duration<float, std::milli>(
                            wall_end - wall_start).count();
 
            stats.idle_ms = std::abs(stats.cpu_ms - stats.gpu_ms);
            return stats;
        }

        float run_cpu_partition(const KFInstance& job,
                            int start, int end,
                            int& omp_threads_out, int n_cpu) const {
            
            #pragma omp parallel
            {
                #pragma omp single
                omp_threads_out = omp_get_num_threads();
            }
            
            kfcpu::CpuSlice slice {
            .x0 = job.initial_states,
            .m0   = job.measurements,
            .output         = job.output,
            .filter_count   = n_cpu,
            .step_count     = job.step_count,
            .state_dim      = job.state_dim,
            };

            auto t0 = std::chrono::high_resolution_clock::now();
            kfcpu::kf_launch_cpu(slice); 
            auto t1 = std::chrono::high_resolution_clock::now();
 
            return std::chrono::duration<float, std::milli>(t1 - t0).count();
        }

    void launch_gpu_partition(const KFInstance& job,
                          int gpu_start, int n_gpu,
                          ExStats& stats) {
    kfgpuI::GpuSlice slice {
        .initial_states = job.initial_states + gpu_start * job.state_dim,
        .measurements   = job.measurements   + gpu_start * job.step_count * job.state_dim,
        .output         = job.output         + gpu_start * job.step_count * job.state_dim,
        .filter_count   = n_gpu,
        .step_count     = job.step_count,
        .state_dim      = job.state_dim,
    };

    gpu_bufs_ = kfgpuI::alloc_dbuffers(slice);

    cudaEventRecord(gpu_ctx_.ev_start(), gpu_ctx_.stream());
    kfgpuI::kf_launch_gpu(slice, gpu_bufs_, gpu_ctx_.stream());
    cudaEventRecord(gpu_ctx_.ev_stop(), gpu_ctx_.stream());

    kfgpuI::free_dbuffers(gpu_bufs_);
    }
    private:
        SchedulerConfig              cfg_;
        GpuContext                   gpu_ctx_;
        kfgpuI::GpuDBuffers           gpu_bufs_;
        bool                         gpu_only_;
};

template<typename T>
struct PinnedAllocator {
    using value_type = T;
 
    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        if (cudaMallocHost(&ptr, n * sizeof(T)) != cudaSuccess)
            throw std::bad_alloc();
        return ptr;
    }
 
    void deallocate(T* ptr, std::size_t) {
        cudaFreeHost(ptr);
    }
};
template<typename T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;

}