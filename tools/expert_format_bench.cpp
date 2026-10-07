// Isolate actual routed expert matmul kernels, selecting buffers as llama.cpp does.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc,char ** argv) {
    if(argc<4||argc>5) return 2;
    const auto type=(ggml_type)std::atoi(argv[1]);
    const int iterations=std::atoi(argv[3]);
    if(iterations<10) return 2;
    auto backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,12);
    const auto device=ggml_backend_get_device(backend);
    ggml_init_params params={ggml_tensor_overhead()*24+ggml_graph_overhead(),nullptr,true};
    auto weights_ctx=ggml_init(params),ctx=ggml_init(params);
    auto w=ggml_new_tensor_3d(weights_ctx,type,640,2560,512);
    auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,640,8,1);
    auto ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,8,1);
    auto y=ggml_mul_mat_id(ctx,w,x,ids);
    auto buft=ggml_backend_get_default_buffer_type(backend);
    auto get_extra=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(ggml_backend_cpu_reg(),"ggml_backend_dev_get_extra_bufts");
    if(get_extra && (argc==4 || std::string(argv[4])!="native")) {
        for(auto candidates=get_extra(device);candidates&&*candidates;++candidates) {
            auto probe=ggml_backend_buft_alloc_buffer(*candidates,0);
            w->buffer=probe;x->buffer=probe;ids->buffer=probe;
            const bool supported=ggml_backend_dev_supports_op(device,y);
            w->buffer=nullptr;x->buffer=nullptr;ids->buffer=nullptr;
            ggml_backend_buffer_free(probe);
            if(supported) {buft=*candidates;break;}
        }
    }
    auto weights=ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx,buft);
    auto operands=ggml_backend_alloc_ctx_tensors(ctx,backend);
    std::vector<char> packed(ggml_nbytes(w));
    std::ifstream file(argv[2],std::ios::binary);file.read(packed.data(),packed.size());
    if(!file) return 3;
    ggml_backend_tensor_set(w,packed.data(),0,packed.size());
    packed.clear();packed.shrink_to_fit();
    std::mt19937 rng(1729);std::normal_distribution<float> normal;
    std::vector<float> input(640*8);
    for(auto & value:input) value=normal(rng);
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    auto allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!ggml_gallocr_alloc_graph(allocator,graph)) return 4;
    std::vector<float> result(2560*8);
    for(int group=0;group<5;++group) {
        double elapsed=0;
        for(int step=0;step<iterations+10;++step) {
            int32_t routes[8];
            for(int j=0;j<8;++j) routes[j]=(step*37+group*103+j*61)%512;
            ggml_backend_tensor_set(ids,routes,0,sizeof(routes));
            const auto start=std::chrono::steady_clock::now();
            if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 5;
            const auto end=std::chrono::steady_clock::now();
            if(step>=10) elapsed+=std::chrono::duration<double,std::milli>(end-start).count();
        }
        ggml_backend_tensor_get(y,result.data(),0,result.size()*sizeof(float));
        for(auto value:result) if(!std::isfinite(value)) return 6;
        std::printf("{\"group\":%d,\"milliseconds_per_routed_matmul\":%.6f,\"iterations\":%d,\"threads\":12,\"experts_per_token\":8,\"buffer\":\"%s\"}\n",group,elapsed/iterations,iterations,ggml_backend_buft_name(buft));
    }
    ggml_gallocr_free(allocator);ggml_backend_buffer_free(operands);ggml_backend_buffer_free(weights);
    ggml_free(ctx);ggml_free(weights_ctx);ggml_backend_free(backend);return 0;
}
