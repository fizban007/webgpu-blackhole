class DiskVisualization {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.device = null;
        this.context = null;
        this.computePipeline = null;
        this.renderPipeline = null;
        this.uniformBuffer = null;
        this.computeBindGroup = null;
        this.renderBindGroup = null;
        this.outputTexture = null;
        this.sampler = null;
        
        this.diskRadius = 20.0;
        this.innerRadius = 5.0;
        this.observerDistance = 35.0;
        this.blackHoleMass = 1.0;
        this.blackHoleSpin = 0.0;
        
        this.camera = {
            theta: 0,
            phi: Math.PI / 3,
            radius: this.observerDistance
        };
        
        this.mouseState = {
            isDown: false,
            lastX: 0,
            lastY: 0
        };
        
        this.init();
    }
    
    updateUI() {
        document.getElementById('radius').textContent = this.diskRadius.toFixed(1);
        document.getElementById('innerRadius').textContent = this.innerRadius.toFixed(1);
        document.getElementById('distance').textContent = this.observerDistance.toFixed(1);
        document.getElementById('mass').textContent = this.blackHoleMass.toFixed(1);
        document.getElementById('spin').textContent = this.blackHoleSpin.toFixed(1);
    }
    
    async init() {
        try {
            await this.initWebGPU();
            this.setupEventListeners();
            this.createComputePipeline();
            this.createRenderPipeline();
            this.createBuffersAndTextures();
            this.updateUI();
            this.render();
        } catch (error) {
            console.error('Failed to initialize WebGPU:', error);
            document.body.innerHTML = '<div style="color:white;padding:20px;">WebGPU Error: ' + error.message + '</div>';
            throw error;
        }
    }
    
    async initWebGPU() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }
        
        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu');
        
        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: canvasFormat,
        });
        
        this.resizeCanvas();
    }
    
    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    createComputePipeline() {
        const computeShaderCode = `
            struct Uniforms {
                cameraPos: vec3<f32>,
                padding1: f32,
                diskRadius: f32,
                innerRadius: f32,
                blackHoleMass: f32,
                blackHoleSpin: f32,
                screenWidth: f32,
                screenHeight: f32,
                observerDistance: f32,
                padding2: f32,
                viewMatrix: mat4x4<f32>,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
            
            // Include all the GR ray tracing functions here...
            [GR_FUNCTIONS_PLACEHOLDER]
            
            @compute @workgroup_size(8, 8, 1)
            fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let pixel_coord = vec2<i32>(global_id.xy);
                let screen_size = vec2<i32>(i32(uniforms.screenWidth), i32(uniforms.screenHeight));
                
                if (pixel_coord.x >= screen_size.x || pixel_coord.y >= screen_size.y) {
                    return;
                }
                
                // Block-based rendering
                let blockSize = 16;
                let blockX = (pixel_coord.x / blockSize) * blockSize;
                let blockY = (pixel_coord.y / blockSize) * blockSize;
                
                let centerX = f32(blockX + blockSize / 2);
                let centerY = f32(screen_size.y - blockY - blockSize / 2);
                
                let blockScreenPos = vec2<f32>(
                    (centerX / uniforms.screenWidth) * 2.0 - 1.0,
                    (centerY / uniforms.screenHeight) * 2.0 - 1.0
                );
                
                let aspectRatio = uniforms.screenWidth / uniforms.screenHeight;
                let ndc = vec2<f32>(blockScreenPos.x * aspectRatio, blockScreenPos.y);
                
                let fov = 0.785398;
                let rayDirLocal = normalize(vec3<f32>(ndc.x * tan(fov/2.0), ndc.y * tan(fov/2.0), -1.0));
                
                let viewMatrixInv = transpose(uniforms.viewMatrix);
                let rayDir = normalize((viewMatrixInv * vec4<f32>(rayDirLocal, 0.0)).xyz);
                
                let grHit = traceGeodesic(uniforms.cameraPos, rayDir, uniforms.blackHoleSpin, uniforms.blackHoleMass, uniforms.diskRadius, uniforms.innerRadius, uniforms.observerDistance * 2.0);
                
                var color: vec4<f32>;
                if (grHit) {
                    color = vec4<f32>(1.0, 0.2, 0.2, 1.0);
                } else {
                    color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }
                
                textureStore(outputTexture, pixel_coord, color);
            }
        `;
        
        // Read the GR functions from the original file and insert them
        // For now, let's keep the original approach but plan the compute shader migration
        console.log('Compute pipeline creation planned - continuing with fragment shader for now');
    }
    
    createRenderPipeline() {
        // Keep existing render pipeline for now
        // Implementation continues with the fragment shader approach
    }
    
    // Rest of the class implementation...
}

// Initialize when page loads
window.addEventListener('load', () => {
    new DiskVisualization();
});