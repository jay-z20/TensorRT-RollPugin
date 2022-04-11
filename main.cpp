#include "logging.h"
#include "utilsn.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include<fstream>
#include<iostream>
#include<map>
#include<sstream>
#include<vector>
#include<chrono>

using namespace nvinfer1;

static Logger gLogger;


const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME  = "output";

static const int INPUT_H = 4; //
static const int INPUT_W = 5; //

cudaStream_t m_cudaStream;
IExecutionContext *m_context;

std::vector<int> shift_sizes;
std::vector<int> dims;

ITensor* roll(INetworkDefinition* m_network, ITensor* input) {
	auto creator = getPluginRegistry()->getPluginCreator("rollLayer_TRT", "1");
	PluginField pField[4];
	pField[0].data = shift_sizes.data();
	pField[0].length = shift_sizes.size();
	pField[0].type = PluginFieldType::kINT32;
	pField[0].name = "shift_sizes";

	pField[1].data = dims.data();
	pField[1].length = dims.size();
	pField[1].type = PluginFieldType::kINT32;
	pField[1].name = "dims";
	std::vector<int> strids;
	strids.push_back(1);
	int s = 1;
	for (int i = input->getDimensions().nbDims - 1; i > 0; i--) {
		s *= input->getDimensions().d[i];
		strids.push_back(s);
	}
	std::reverse(strids.begin(), strids.end()); // 翻转 对于3x4x5 的矩阵 strids:[20,5,1]

	pField[2].data = strids.data();
	pField[2].length = strids.size();
	pField[2].type = PluginFieldType::kINT32;
	pField[2].name = "strids";

	std::vector<int>shapes;
	
	for (int i = 0; i < input->getDimensions().nbDims; i++)
		shapes.push_back(input->getDimensions().d[i]);

	pField[3].data = shapes.data();
	pField[3].length = shapes.size();
	pField[3].type = PluginFieldType::kINT32;
	pField[3].name = "shapes";

	PluginFieldCollection pluginData;
	pluginData.nbFields = 4;
	pluginData.fields = pField;
	IPluginV2 *pluginObj = creator->createPlugin("Roll", &pluginData);
	ITensor* inputTensor[] = { input };
	auto roll_p = m_network->addPluginV2(inputTensor, 1, *pluginObj);
	return roll_p->getOutput(0);
}



ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder,IBuilderConfig *config,
	DataType dt){
	INetworkDefinition *network = builder->createNetworkV2(0U);
	ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3,INPUT_H,INPUT_W });
	assert(data);
	// std::vector<int>shift_sizes{ 1,1,1 };
	// std::vector<int>dims{ 0,1,2 };
	ITensor* out = roll(network, data);
	out->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*out);
	builder->setMaxBatchSize(1);
	config->setMaxWorkspaceSize(16 * (1 << 20));

	std::cout << "Building engine, please wait for a while..." << std::endl;
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "Build engine successfully!" << std::endl;
	network->destroy();
	return engine;
}

void APITOModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
	IBuilder* builder = createInferBuilder(gLogger);
	IBuilderConfig* config = builder->createBuilderConfig();
	ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
	std::cout << "APITOModel" << std::endl;
	if (engine==nullptr)
	{
		std::cout << "engine nullprt=====" << std::endl;
	}
	assert(engine != nullptr);
	// 序列化
	(*modelStream) = engine->serialize();
	engine->destroy();
	builder->destroy();
	config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input,
	float* output) {
	CUDA_CHECK(cudaMemcpyAsync(buffers[0],input,3*INPUT_H*INPUT_W*sizeof(float),cudaMemcpyHostToDevice));
	context.enqueue(1, buffers, stream, nullptr);
	CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], 3 * INPUT_H*INPUT_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
}

void inline printData(float* mat) {
	int index, index1 = 0;
	for (int i = 0; i < 3; i++) {
		index = i * INPUT_H*INPUT_W;
		std::cout << "[";
		for (int j = 0; j < INPUT_H; j++)
		{
			index1 = index + j * INPUT_W;
			std::cout << "[";
			for (int k = 0; k < INPUT_W; k++)
			{
				std::cout << mat[index1 + k] << " ";
			}
			std::cout << "]" << std::endl;
		}
		std::cout << "]" << std::endl;
	}
}

void parse(int argc, char** argv) {
	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-shift") {
			for (int j = i+1; j < argc; j++)
			{
				string arg = argv[j];
				if (arg[0]=='-')
					break;
				shift_sizes.push_back(stoi(arg));
			}
		}
		else if (std::string(argv[i]) == "-dims") {
			for (int j = i + 1; j < argc; j++)
			{
				string arg = argv[j];
				if (arg[0] == '-')
					break;
				dims.push_back(stoi(arg));
			}
		}

	}
}

int main(int argc, char** argv) {
	
	// 构建 engine
	std::cout << "save engine: roll.engine" << std::endl;
	parse(argc, argv);

	if (std::string(argv[1]) == "-s") {

		if (shift_sizes.size() == 0)
		{
			shift_sizes = { 1,1 };
			dims = { 1,2 };
		}
		IHostMemory* modelStream{ nullptr };
		APITOModel(1, &modelStream);
		if (modelStream == nullptr)
		{
			std::cout << "error!!" << std::endl;
		}

		std::ofstream p("roll.engine", std::ios::binary);
		if (!p) {
			std::cerr << "could not open plan output file" << std::endl;
			return -1;
		}
		p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
		// createEng(wts,eng);
		modelStream->destroy();
		return 0;
	}
	
	// 推理过程
	std::ifstream file("roll.engine", std::ios::binary);
	if (!file.good())
	{
		std::cerr << "read error!!" << std::endl;
		return -1;
	}
	char *trtModelStream = nullptr;
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	trtModelStream = new char[size];
	assert(trtModelStream);
	file.read(trtModelStream, size);
	file.close();


	
	// 测试数据 3x4x5
	static float data[1 * 3 * INPUT_H*INPUT_W];
	static float out[1 * 3 * INPUT_H*INPUT_W];
	for (int i = 0; i < 3 * INPUT_H*INPUT_W; i++)
		data[i] = i;


	IRuntime* runtime = createInferRuntime(gLogger);

	assert(runtime != nullptr);
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);
	delete[] trtModelStream;
	const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

	assert(inputIndex == 0);
	assert(outputIndex == 1);
	void* buffers[2];
	CUDA_CHECK(cudaMalloc(&buffers[0], 3 * INPUT_H*INPUT_W * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&buffers[1], 3 * INPUT_H*INPUT_W * sizeof(float)));
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	// run inference
	auto start = std::chrono::system_clock::now();
	std::cout << "Inference!!" << std::endl;
	doInference(*context, stream, buffers, data, out);
	std::cout << "====== Input Data 3x4x5: ======" << std::endl;
	printData(data);

	std::cout << "Roll argv:" << std::endl;
	std::cout << "\tshift size: ";
	for (int i = 0; i < shift_sizes.size(); i++)
		std::cout << shift_sizes[i] << " ";
	std::cout << std::endl;

	std::cout << "\tdims: ";
	for (int i = 0; i < dims.size(); i++)
		std::cout << dims[i] << " ";
	std::cout << std::endl;

	std::cout << "====== Roll Op Output Data 3x4x5: ======" << std::endl;
	printData(out);

	cudaStreamDestroy(stream);
	CUDA_CHECK(cudaFree(buffers[0]));
	CUDA_CHECK(cudaFree(buffers[1]));
	context->destroy();
	engine->destroy();
	runtime->destroy();
	return 0;
}

