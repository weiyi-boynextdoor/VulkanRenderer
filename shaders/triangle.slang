struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

struct VertexInput
{
    float3 position : POSITION;
    float3 color : COLOR0;
};

struct CameraData
{
    float4x4 modelViewProj;
};

[[vk::push_constant]]
ConstantBuffer<CameraData> gCamera;

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = mul(gCamera.modelViewProj, float4(input.position, 1.0));
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
