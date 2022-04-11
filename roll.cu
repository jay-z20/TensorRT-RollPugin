#include"roll.h"
#include<algorithm>
#include<math.h>
#include<device_launch_parameters.h>

__host__ __forceinline__ void copyData(vector<int>&v, int*& tmp, int len) {
	for (int i = 0; i < len; i++)
		v[i] = tmp[i];
}

namespace nvinfer1 {
	roll::roll(const std::vector<int>& vshift_sizes, const std::vector<int>& vdims,
		const std::vector<int>& vstrids, const std::vector<int>& vshapes) {
		N = vshift_sizes.size();
		sN = vshapes.size();
		rshift_sizes = vshift_sizes;
		rdims = vdims;
		rstrids = vstrids;
		rshapes = vshapes;
		CUDA_CHECK(cudaMalloc(&shifts, vshift_sizes.size() * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(shifts, vshift_sizes.data(), vshift_sizes.size() * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&dims, vdims.size() * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(dims, vdims.data(), vdims.size() * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&strides, vstrids.size() * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(strides, vstrids.data(), vstrids.size() * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&shapes, vshapes.size() * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(shapes, vshapes.data(), vshapes.size() * sizeof(int), cudaMemcpyHostToDevice));
	}
	roll::~roll() {
		CUDA_CHECK(cudaFree(shifts));
		CUDA_CHECK(cudaFree(dims));
		CUDA_CHECK(cudaFree(strides));
		CUDA_CHECK(cudaFree(shapes));
	}
	// 反序列化
	roll::roll(const void* data, size_t length) {
		const char* d = reinterpret_cast<const char*>(data), *a = d;
		Tn::read(d, mInputSize);
		Tn::read(d, N);
		Tn::read(d, sN);
		int size = (int)N * sizeof(int);
		rshift_sizes.resize(N);
		memcpy(rshift_sizes.data(), d, size);

		d += size;
		rdims.resize(N);
		memcpy(rdims.data(), d, size);

		std::cout << std::endl;
		d += size;

		size = (int)sN * sizeof(int);
		rstrids.resize(sN);
		memcpy(rstrids.data(), d, size);
		d += size;
		rshapes.resize(sN);
		memcpy(rshapes.data(), d, size);
		d += size;

		CUDA_CHECK(cudaMalloc(&shifts, rshift_sizes.size() * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(shifts, rshift_sizes.data(), rshift_sizes.size() * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&dims, rdims.size() * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(dims, rdims.data(), rdims.size() * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&strides, rstrids.size() * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(strides, rstrids.data(), rstrids.size() * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMalloc(&shapes, rshapes.size() * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(shapes, rshapes.data(), rshapes.size() * sizeof(int), cudaMemcpyHostToDevice));

		assert(d == a + length);
	}
	// 序列化
	void roll::serialize(void* buffer) const {
		char* d = static_cast<char*>(buffer), *a = d;
		Tn::write(d, mInputSize);
		Tn::write(d, N);
		Tn::write(d, sN);
		int size = rshift_sizes.size() * sizeof(int);
		memcpy(d, rshift_sizes.data(), size);
		d += size;
		size = rdims.size() * sizeof(int);
		memcpy(d, rdims.data(), size);
		d += size;
		size = rstrids.size() * sizeof(int);
		memcpy(d, rstrids.data(), size);
		d += size;
		size = rshapes.size() * sizeof(int);
		memcpy(d, rshapes.data(), size);
		d += size;
		assert(d == a + getSerializationSize());
	}
	size_t roll::getSerializationSize() const {
		return sizeof(mInputSize) + sizeof(N) + sizeof(sN) + rshift_sizes.size() * sizeof(int) +
			rdims.size() * sizeof(int)+ rstrids.size() * sizeof(int)+
			rshapes.size() * sizeof(int);
	}

	int roll::initialize() {
		return 0;
	}

	Dims roll::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
		assert(nbInputDims == 1);
		Dims outputDims;
		outputDims.nbDims = inputs[0].nbDims;
		for (int i = 0; i < inputs[0].nbDims; i++)
		{
			outputDims.d[i] = inputs[0].d[i];
		}
		return outputDims;
	}

	// Set plugin namespace
	void roll::setPluginNamespace(const char* pluginNamespace)
	{
		mPluginNamespace = pluginNamespace;
	}

	const char* roll::getPluginNamespace() const
	{
		return mPluginNamespace;
	}

	// Return the DataType of the plugin output at the requested index
	DataType roll::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
	{
		return DataType::kFLOAT;
	}

	// Return true if output tensor is broadcast across a batch.
	bool roll::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
	{
		return false;
	}

	// Return true if plugin can use input that is broadcast across batch without replication.
	bool roll::canBroadcastInputAcrossBatch(int inputIndex) const
	{
		return false;
	}

	void roll::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
	{

		mInputSize = 1;
		for (int i = 0; i < in[0].dims.nbDims; i++) {
			mInputSize *= in[0].dims.d[i];
		}
	}

	// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
	void roll::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
	{
	}

	// Detach the plugin object from its execution context.
	void roll::detachFromContext() {}

	const char* roll::getPluginType() const
	{
		return "rollLayer_TRT";
	}

	const char* roll::getPluginVersion() const
	{
		return "1";
	}

	void roll::destroy()
	{
		delete this;
	}

	// Clone the plugin
	IPluginV2IOExt* roll::clone() const
	{
		roll *p = new roll(rshift_sizes, rdims, rstrids, rshapes);
		p->setPluginNamespace(mPluginNamespace);
		p->setInputSize(mInputSize);
		return p;
	}

	__global__ void rollKernel(const float *in, float *out, int size,int Ndims,const int* rshift,
		const int* rdims,const int* rstrids,const int* rshapes) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= size) return;
		int new_dim = 0;
		int new_idx = idx;;
		#pragma unroll
		for (size_t i = 0; i < Ndims; i++)
		{
			int ind = rdims[i];
			new_dim = (idx / rstrids[ind])%rshapes[ind]+rshift[i];
			//需要考虑 越界循环
			if (new_dim>=rshapes[ind]) 
				new_idx += (rshift[i] - rshapes[ind])*rstrids[ind];
			else
				new_idx += rshift[i]*rstrids[ind];
		}
		out[new_idx] = in[idx];
	}
	void roll::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize) {
		int numElem = batchSize * mInputSize;
		rollKernel << <(numElem + mThreadCount - 1) / mThreadCount, mThreadCount >> > 
			(inputs[0], output, numElem, N, (const int*)shifts, (const int*)dims, (const int*)strides, (const int*)shapes);
	}

	int roll::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
	{
		forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
		return 0;
	}

	PluginFieldCollection rollCreator::mFC{};
	std::vector<PluginField> rollCreator::mPluginAttributes;

	rollCreator::rollCreator()
	{
		mPluginAttributes.clear();
		mFC.nbFields = mPluginAttributes.size();
		mFC.fields = mPluginAttributes.data();
	}

	const char* rollCreator::getPluginName() const
	{
		return "rollLayer_TRT";
	}

	const char* rollCreator::getPluginVersion() const
	{
		return "1";
	}

	const PluginFieldCollection* rollCreator::getFieldNames()
	{
		return &mFC;
	}


	IPluginV2IOExt* rollCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
	{
		const PluginField* fields = fc->fields;
		std::vector<int> vshift_sizes, vdims,
			vstrids, vshapes;

		for (int i = 0; i < fc->nbFields; i++)
		{
			int* tmp = (int*)(fields[i].data);
			if (strcmp(fields[i].name, "shift_sizes") == 0) {
				for (int j = 0; j < fields[i].length; j++)
					vshift_sizes.push_back(tmp[j]);
			}
			else if (strcmp(fields[i].name, "dims") == 0) {
				for (int j = 0; j < fields[i].length; j++) {
					vdims.push_back(tmp[j]);
				}
			}
			else if (strcmp(fields[i].name, "strids") == 0) {
				for (int j = 0; j < fields[i].length; j++)
					vstrids.push_back(tmp[j]);
			}
			else {
				for (int j = 0; j < fields[i].length; j++)
					vshapes.push_back(tmp[j]);
			}
		}
		
		assert(vshift_sizes.size() > 0);
		assert(vshift_sizes.size() == vdims.size());
		roll* obj = new roll(vshift_sizes, vdims, vstrids, vshapes);
		obj->setPluginNamespace(mNamespace.c_str());
		return obj;
	}

	IPluginV2IOExt* rollCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
	{
		// This object will be deleted when the network is destroyed, which will
		roll* obj = new roll(serialData, serialLength);
		obj->setPluginNamespace(mNamespace.c_str());
		return obj;
	}
};