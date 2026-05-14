namespace kfcpu {

struct CpuSlice {
    const double* x0;
    const double* m0; 
    double*       output;      

    int filter_count;           
    int step_count;
    int state_dim;
};

void kf_launch_cpu(CpuSlice& slice);
}