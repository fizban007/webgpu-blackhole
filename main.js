class DiskVisualization {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.uniformBuffer = null;
        this.bindGroup = null;
        
        this.diskRadius = 10.0;
        this.innerRadius = 4.0;
        this.observerDistance = 30.0;
        
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
    }
    
    async init() {
        try {
            await this.initWebGPU();
            this.setupEventListeners();
            this.createRenderPipeline();
            this.createUniformBuffer();
            this.updateUI();
            this.render();
        } catch (error) {
            console.error('Failed to initialize WebGPU:', error);
            document.body.innerHTML = '<div style="color:white;padding:20px;">WebGPU Error: ' + error.message + '</div>';
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
    
    createRenderPipeline() {
        const vertexShaderCode = `
            @vertex
            fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
                var pos = array<vec2<f32>, 3>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>( 3.0, -1.0),
                    vec2<f32>(-1.0,  3.0)
                );
                return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
            }
        `;
        
        const fragmentShaderCode = `
            struct Uniforms {
                cameraPos: vec3<f32>,
                padding1: f32,
                diskRadius: f32,
                innerRadius: f32,
                screenWidth: f32,
                screenHeight: f32,
                viewMatrix: mat4x4<f32>,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            fn rayDiskIntersection(rayOrigin: vec3<f32>, rayDir: vec3<f32>, diskRadius: f32, innerRadius: f32) -> bool {
                if (abs(rayDir.z) < 0.0001) {
                    return false;
                }
                
                let t = -rayOrigin.z / rayDir.z;
                if (t <= 0.0) {
                    return false;
                }
                
                let hitPoint = rayOrigin + t * rayDir;
                let distanceFromCenter = length(hitPoint.xy);
                
                return distanceFromCenter <= diskRadius && distanceFromCenter >= innerRadius;
            }
            
            @fragment
            fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
                let screenPos = vec2<f32>(
                    (fragCoord.x / uniforms.screenWidth) * 2.0 - 1.0,
                    1.0 - (fragCoord.y / uniforms.screenHeight) * 2.0
                );
                
                let aspectRatio = uniforms.screenWidth / uniforms.screenHeight;
                let ndc = vec2<f32>(screenPos.x * aspectRatio, screenPos.y);
                
                let fov = 0.785398;
                let rayDirLocal = normalize(vec3<f32>(ndc.x * tan(fov/2.0), ndc.y * tan(fov/2.0), -1.0));
                
                let viewMatrixInv = transpose(uniforms.viewMatrix);
                let rayDir = normalize((viewMatrixInv * vec4<f32>(rayDirLocal, 0.0)).xyz);
                
                if (rayDiskIntersection(uniforms.cameraPos, rayDir, uniforms.diskRadius, uniforms.innerRadius)) {
                    return vec4<f32>(1.0, 0.2, 0.2, 1.0);
                } else {
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }
            }
        `;
        
        const vertexShader = this.device.createShaderModule({
            code: vertexShaderCode,
        });
        
        const fragmentShader = this.device.createShaderModule({
            code: fragmentShaderCode,
        });
        
        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: vertexShader,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: fragmentShader,
                entryPoint: 'fs_main',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                }],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });
    }
    
    createUniformBuffer() {
        const uniformBufferSize = 4 * 4 + 4 * 4 + 4 * 16;
        this.uniformBuffer = this.device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        
        this.bindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer,
                },
            }],
        });
    }
    
    updateCamera() {
        const x = this.camera.radius * Math.sin(this.camera.phi) * Math.cos(this.camera.theta);
        const y = this.camera.radius * Math.sin(this.camera.phi) * Math.sin(this.camera.theta);
        const z = this.camera.radius * Math.cos(this.camera.phi);
        
        const eye = [x, y, z];
        const center = [0, 0, 0];
        const up = [0, 0, 1];
        
        const viewMatrix = this.lookAt(eye, center, up);
        
        const uniformData = new Float32Array(4 + 4 + 16);
        
        uniformData[0] = x;
        uniformData[1] = y;
        uniformData[2] = z;
        uniformData[3] = 0;
        
        uniformData[4] = this.diskRadius;
        uniformData[5] = this.innerRadius;
        uniformData[6] = this.canvas.width;
        uniformData[7] = this.canvas.height;
        
        for (let i = 0; i < 16; i++) {
            uniformData[8 + i] = viewMatrix[i];
        }
        
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }
    
    lookAt(eye, center, up) {
        const f = this.normalize(this.subtract(center, eye));
        const s = this.normalize(this.cross(f, up));
        const u = this.cross(s, f);
        
        return new Float32Array([
            s[0], u[0], -f[0], 0,
            s[1], u[1], -f[1], 0,
            s[2], u[2], -f[2], 0,
            -this.dot(s, eye), -this.dot(u, eye), this.dot(f, eye), 1
        ]);
    }
    
    perspective(fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);
        
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0
        ]);
    }
    
    normalize(v) {
        const length = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return [v[0] / length, v[1] / length, v[2] / length];
    }
    
    subtract(a, b) {
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    }
    
    cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }
    
    dot(a, b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    
    setupEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => {
            this.mouseState.isDown = true;
            this.mouseState.lastX = e.clientX;
            this.mouseState.lastY = e.clientY;
        });
        
        this.canvas.addEventListener('mouseup', () => {
            this.mouseState.isDown = false;
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.mouseState.isDown) return;
            
            const deltaX = e.clientX - this.mouseState.lastX;
            const deltaY = e.clientY - this.mouseState.lastY;
            
            this.camera.theta += deltaX * 0.01;
            this.camera.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.camera.phi - deltaY * 0.01));
            
            this.mouseState.lastX = e.clientX;
            this.mouseState.lastY = e.clientY;
            
            console.log(`Camera: theta=${this.camera.theta.toFixed(2)}, phi=${this.camera.phi.toFixed(2)}`);
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const deltaDistance = e.deltaY * 0.05;
            this.camera.radius = Math.max(10.0, Math.min(400.0, this.camera.radius + deltaDistance));
            this.observerDistance = this.camera.radius;
            this.updateUI();
        });
        
        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });
    }
    
    render() {
        this.updateCamera();
        
        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();
        
        const renderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        };
        
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.renderPipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        passEncoder.draw(3, 1, 0, 0);
        passEncoder.end();
        
        this.device.queue.submit([commandEncoder.finish()]);
        
        requestAnimationFrame(() => this.render());
    }
}

new DiskVisualization();
