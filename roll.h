#pragma once

#include<vector>
#include<string>
#include<NvInfer.h>
#include<NvInferPlugin.h>
#include"NvInferRuntimeCommon.h"
#include"utilsn.h"
#include<assert.h>



namespace nvinfer1 {

	class roll :public IPluginV2IOExt
	{
	public:
		roll(const std::vector<int>& vshift_sizes, const std::vector<int>& vdims,
			const std::vector<int>& vstrids, const std::vector<int>& vshapes);
		roll(const void* data, size_t length);
		~roll();
		int getNbOutputs() const override
		{
			return 1;
		}

		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
		int initialize() override;
		virtual void terminate() override {};
		virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }
		virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
		virtual size_t getSerializationSize() const override;
		virtual void serialize(void* buffer) const override;

		bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
			return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
		}

		const char* getPluginType() const override;
		const char* getPluginVersion() const override;
		void destroy() override;
		IPluginV2IOExt* clone() const override;
		void setPluginNamespace(const char* pluginNamespace) override;
		const char* getPluginNamespace() const override;
		DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
		bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
		bool canBroadcastInputAcrossBatch(int inputIndex) const override;
		void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;
		void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;
		void detachFromContext() override;

		void setInputSize(int s) {
			mInputSize = s;
		}

	private:
		void forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize = 1);
		int mThreadCount = 256;
		int N,sN; // roll 维度数量
		int mInputSize;
		const char* mPluginNamespace;
		std::vector<int>rshift_sizes; // 用于 clone
		std::vector<int>rdims;
		std::vector<int>rstrids;
		std::vector<int>rshapes;
		void **shifts;
		void **dims;
		void **shapes;
		void **strides;
	};

	class rollCreator : public IPluginCreator
	{
	public:
		rollCreator();
		~rollCreator() override = default;
		const char* getPluginName() const override;
		const char* getPluginVersion() const override;
		const PluginFieldCollection* getFieldNames() override;
		IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;
		IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

		void setPluginNamespace(const char* libNamespace) override
		{
			mNamespace = libNamespace;
		}

		const char* getPluginNamespace() const override
		{
			return mNamespace.c_str();
		}

	private:
		std::string mNamespace;
		static PluginFieldCollection mFC;
		static std::vector<PluginField> mPluginAttributes;
	};
	REGISTER_TENSORRT_PLUGIN(rollCreator);
};
