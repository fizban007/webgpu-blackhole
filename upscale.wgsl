@group(0) @binding(0) var lowResTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

struct Uniforms {
    screenWidth: f32,
    screenHeight: f32,
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    // Use full screen dimensions for UV calculation
    let uv = fragCoord.xy / vec2f(uniforms.screenWidth, uniforms.screenHeight);
    
    // Sample the low-resolution texture with linear interpolation
    return textureSample(lowResTexture, textureSampler, uv);
}