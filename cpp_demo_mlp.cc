#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/runtime/relax_vm/vm.h>
#include <tvm/runtime/container/adt.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace tvm;
using namespace tvm::runtime;
using namespace relax_vm;

const int DATA_LINES_NUM = 10000;

float X_data[DATA_LINES_NUM][784], 
      LABEL_data[DATA_LINES_NUM][10],
      BATCH_W0_ADJOINT[784*128],
      BATCH_B0_ADJOINT[128],
      BATCH_W1_ADJOINT[128*10],
      BATCH_B1_ADJOINT[10];

int main() {
    /* 
        Load Module to ex
        Notice: .so should be add to LD_LIBRARY_PATH or use absolute path so that dlopen can find it
    */
    Module mod = Module::LoadFromFile("mlp_mod.so");

    /*
        Build relax virtual machine
    */
    Module vm = mod.GetFunction("vm_load_executable")();
    vm.GetFunction("vm_initialization")(1, 0, 2); // 1, 0, 2 (device:cpu(0), alloca type:default)

    /* 
        Relax function to be executed 
    */
    auto pred_func = vm.GetFunction("main_pred");
    auto train_func = vm.GetFunction("main_adjoint");

    /* 
        Preprae Data 
        Init params and load x, label from processed data package
    */
    auto X = NDArray::Empty({1, 784}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto W0 = NDArray::Empty({784, 128}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto B0 = NDArray::Empty({128}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto W1 = NDArray::Empty({128, 10}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto B1 = NDArray::Empty({10}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto LABEL = NDArray::Empty({1, 10}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto pX = static_cast<float*>(X->data);
    auto pW0 = static_cast<float*>(W0->data);
    auto pB0 = static_cast<float*>(B0->data);
    auto pW1 = static_cast<float*>(W1->data);
    auto pB1 = static_cast<float*>(B1->data);
    auto pLABEL = static_cast<float*>(LABEL->data);

    /* Data Loader */
    ifstream X_file = ifstream("data/fanshionMNIST_data_x", ios::in | ios::binary);
    ifstream LABEL_file = ifstream("data/fanshionMNIST_data_label", ios::in | ios::binary);
    X_file.read(reinterpret_cast<char*>(X_data), 10000*784*4 /*bytes*/);
    LABEL_file.read(reinterpret_cast<char*>(LABEL_data), 10000*10*4 /*bytes*/);
    X_file.close();
    LABEL_file.close();

    /* Init params */
    srand(time(NULL));
    int N = 100000;
    for (int i = 0; i < 784*128; ++i) pW0[i] = 0.08 * (rand()%N-N/2.0)/(N/2.0);
    for (int i = 0; i < 128; ++i) pB0[i] = 0.2 * (rand()%N-N/2.0)/(N/2.0);
    for (int i = 0; i < 128*10; ++i) pW1[i] = 0.08 * (rand()%N-N/2.0)/(N/2.0);
    for (int i = 0; i < 10; ++i) pB1[i] = 0.2 * (rand()%N-N/2.0)/(N/2.0);

    /*
        Training Loop
    */
    float learning_rate = 0.08, loss = 0;
    int batch_size = 64, total_test = 0, correct_test = 0;

    for (int i = 0; i < DATA_LINES_NUM; ++i) {
        /* load X and LABEL for this epoch */
        for (int j = 0; j < 784; ++j) pX[j] = X_data[i][j];
        for (int j = 0; j < 10; ++j) pLABEL[j] = LABEL_data[i][j];

        if (i >= 9000) {
            /* Test */
            total_test++;
            NDArray OUT = pred_func(X, W0, B0, W1, B1);
            auto pOUT = static_cast<float*>(OUT->data);
            int argmax_out = 0, argmax_label = 0;
            for (int j = 1; j < 10; ++j) argmax_out = (pOUT[argmax_out] < pOUT[j]) ? j : argmax_out, argmax_label = (pLABEL[argmax_label] < pLABEL[j]) ? j : argmax_label;
            correct_test += (argmax_out == argmax_label);
            continue;
        }

        /* Get return */
        ADT ret = train_func(X, W0, B0, W1, B1, LABEL);

        auto LOSS = Downcast<NDArray>(ret[0]);
        auto adjoints = Downcast<ADT>(ret[1]);

        auto W0_ADJOINT = Downcast<NDArray>(adjoints[0]);
        auto B0_ADJOINT = Downcast<NDArray>(adjoints[1]);
        auto W1_ADJOINT = Downcast<NDArray>(adjoints[2]);
        auto B1_ADJOINT = Downcast<NDArray>(adjoints[3]);

        auto pLOSS = static_cast<float*>(LOSS->data);
        auto pW0_ADJOINT = static_cast<float*>(W0_ADJOINT->data);
        auto pB0_ADJOINT = static_cast<float*>(B0_ADJOINT->data);
        auto pW1_ADJOINT = static_cast<float*>(W1_ADJOINT->data);
        auto pB1_ADJOINT = static_cast<float*>(B1_ADJOINT->data);

        for (int j = 0; j < 784*128; ++j) BATCH_W0_ADJOINT[j] += pW0_ADJOINT[j];
        for (int j = 0; j < 128; ++j) BATCH_B0_ADJOINT[j] += pB0_ADJOINT[j];
        for (int j = 0; j < 128*10; ++j) BATCH_W1_ADJOINT[j] += pW1_ADJOINT[j];
        for (int j = 0; j < 10; ++j) BATCH_B1_ADJOINT[j] += pB1_ADJOINT[j];
        loss += float(*pLOSS);

        if ((i+1) % batch_size == 0) {
            /* Update params */
            for (int j = 0; j < 784*128; ++j) pW0[j] -= learning_rate * BATCH_W0_ADJOINT[j] / batch_size, BATCH_W0_ADJOINT[j] = 0;
            for (int j = 0; j < 128; ++j) pB0[j] -= learning_rate * BATCH_B0_ADJOINT[j] / batch_size, BATCH_B0_ADJOINT[j] = 0;
            for (int j = 0; j < 128*10; ++j) pW1[j] -= learning_rate * BATCH_W1_ADJOINT[j] / batch_size, BATCH_W1_ADJOINT[j] = 0;
            for (int j = 0; j < 10; ++j) pB1[j] -= learning_rate * BATCH_B1_ADJOINT[j] / batch_size, BATCH_B1_ADJOINT[j] = 0;

            std::cout << "epoch " << i << ": loss = " << loss / batch_size << std::endl;
            loss = 0;
            learning_rate *= 0.995;
        }
    }

    std::cout << "success rate = " << float(correct_test)/total_test<< std::endl;
}