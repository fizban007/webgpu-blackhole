class DiskVisualization {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.uniformBuffer = null;
        this.bindGroup = null;
        
        // Performance monitoring
        this.frameCount = 0;
        this.lastFrameTime = performance.now();
        this.fpsDisplay = null;
        
        this.diskRadius = 20.0;
        this.innerRadius = 5.0;
        this.observerDistance = 100.0;
        this.blackHoleMass = 1.0;
        this.blackHoleSpin = 0.0;
        this.volumetricMode = 1.0; // Start with volumetric mode
        
        this.camera = {
            theta: 0,
            phi: 0.95 * Math.PI / 2,
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
        document.getElementById('spin').textContent = this.blackHoleSpin.toFixed(3);
        
        // Update slider values to match
        document.getElementById('spinSlider').value = this.blackHoleSpin;
        document.getElementById('innerRadiusSlider').value = this.innerRadius;
        document.getElementById('outerRadiusSlider').value = this.diskRadius;
    }
    
    setupPerformanceMonitoring() {
        // Create FPS display element
        this.fpsDisplay = document.createElement('div');
        this.fpsDisplay.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            color: white;
            font-family: monospace;
            font-size: 14px;
            background: rgba(0,0,0,0.5);
            padding: 5px 10px;
            border-radius: 3px;
        `;
        this.fpsDisplay.textContent = 'FPS: --';
        document.body.appendChild(this.fpsDisplay);
    }
    
    updatePerformanceStats() {
        this.frameCount++;
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastFrameTime;
        
        // Update FPS every 60 frames
        if (this.frameCount % 60 === 0) {
            const fps = (60 * 1000 / deltaTime).toFixed(1);
            const frameTime = (deltaTime / 60).toFixed(2);
            this.fpsDisplay.textContent = `FPS: ${fps} (${frameTime}ms/frame)`;
            this.lastFrameTime = currentTime;
        }
    }
    
    async init() {
        try {
            await this.initWebGPU();
            this.setupEventListeners();
            await this.createRenderPipeline();
            this.createUniformBuffer();
            this.updateUI();
            this.setupPerformanceMonitoring();
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
    
    async loadShader(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load shader: ${url}`);
        }
        return await response.text();
    }

    async createRenderPipeline() {
        const vertexShaderCode = await this.loadShader('vertex.wgsl');
        const fragmentShaderCode = await this.loadShader('fragment.wgsl');
        
        let vertexShader, fragmentShader;
        try {
            vertexShader = this.device.createShaderModule({
                code: vertexShaderCode,
            });
            console.log('Vertex shader compiled successfully');
        } catch (error) {
            console.error('Vertex shader compilation failed:', error);
            throw error;
        }
        
        try {
            fragmentShader = this.device.createShaderModule({
                code: fragmentShaderCode,
            });
            console.log('Fragment shader compiled successfully');
        } catch (error) {
            console.error('Fragment shader compilation failed:', error);
            console.log('Fragment shader code:', fragmentShaderCode);
            throw error;
        }
        
        console.log('Fragment shader code length:', fragmentShaderCode.length);
        
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
        // Optimized uniform buffer layout with proper 16-byte alignment
        // Layout: vec3 cameraPos (12 bytes + 4 padding), 5 floats (20 bytes + 12 padding), mat4x4 (64 bytes)
        const uniformBufferSize = 16 + 32 + 64; // 112 bytes total, properly aligned
        this.uniformBuffer = this.device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
        });
        
        this.bindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer,
                    offset: 0,
                    size: uniformBufferSize,
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
        
        // Optimized uniform buffer layout with proper alignment
        const uniformData = new Float32Array(28); // 112 bytes / 4 = 28 floats
        
        // vec3 cameraPos + padding (16 bytes)
        uniformData[0] = x;
        uniformData[1] = y;
        uniformData[2] = z;
        uniformData[3] = 0; // padding
        
        // 8 floats with padding to align to 16-byte boundary (32 bytes)
        uniformData[4] = this.diskRadius;
        uniformData[5] = this.innerRadius;
        uniformData[6] = this.blackHoleMass;
        uniformData[7] = this.blackHoleSpin;
        uniformData[8] = this.canvas.width;
        uniformData[9] = this.canvas.height;
        uniformData[10] = this.observerDistance;
        uniformData[11] = this.volumetricMode;
        
        // mat4x4 viewMatrix (64 bytes) 
        for (let i = 0; i < 16; i++) {
            uniformData[12 + i] = viewMatrix[i];
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
        // Mouse events
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
            
            this.camera.theta -= deltaX * 0.01;
            this.camera.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.camera.phi - deltaY * 0.004));
            
            this.mouseState.lastX = e.clientX;
            this.mouseState.lastY = e.clientY;
            
            console.log(`Camera: theta=${this.camera.theta.toFixed(2)}, phi=${this.camera.phi.toFixed(2)}`);
        });
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                this.mouseState.isDown = true;
                this.mouseState.lastX = e.touches[0].clientX;
                this.mouseState.lastY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                // Store initial pinch distance for zoom
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                this.mouseState.pinchDistance = Math.sqrt(dx * dx + dy * dy);
            }
        });
        
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.mouseState.isDown = false;
            this.mouseState.pinchDistance = null;
        });
        
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            
            if (e.touches.length === 1 && this.mouseState.isDown) {
                // Single touch - rotate camera
                const deltaX = e.touches[0].clientX - this.mouseState.lastX;
                const deltaY = e.touches[0].clientY - this.mouseState.lastY;
                
                this.camera.theta -= deltaX * 0.01;
                this.camera.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.camera.phi - deltaY * 0.004));
                
                this.mouseState.lastX = e.touches[0].clientX;
                this.mouseState.lastY = e.touches[0].clientY;
                
                console.log(`Camera: theta=${this.camera.theta.toFixed(2)}, phi=${this.camera.phi.toFixed(2)}`);
            } else if (e.touches.length === 2 && this.mouseState.pinchDistance) {
                // Two finger pinch - zoom
                const dx = e.touches[0].clientX - e.touches[1].clientX;
                const dy = e.touches[0].clientY - e.touches[1].clientY;
                const currentDistance = Math.sqrt(dx * dx + dy * dy);
                
                const scale = currentDistance / this.mouseState.pinchDistance;
                const deltaDistance = (1 - scale) * this.camera.radius * 0.1;
                
                this.camera.radius = Math.max(10.0, Math.min(400.0, this.camera.radius + deltaDistance));
                this.observerDistance = this.camera.radius;
                this.updateUI();
                
                this.mouseState.pinchDistance = currentDistance;
            }
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const deltaDistance = e.deltaY * 0.05;
            this.camera.radius = Math.max(10.0, Math.min(400.0, this.camera.radius + deltaDistance));
            this.observerDistance = this.camera.radius;
            this.updateUI();
        });

        // Keyboard controls for parameters
        window.addEventListener('keydown', (e) => {
            let updated = false;
            switch(e.key.toLowerCase()) {
                case 's':
                    this.blackHoleSpin = Math.min(0.998, this.blackHoleSpin + 0.05);
                    document.getElementById('spinSlider').value = this.blackHoleSpin;
                    updated = true;
                    break;
                case 'x':
                    this.blackHoleSpin = Math.max(-0.998, this.blackHoleSpin - 0.05);
                    document.getElementById('spinSlider').value = this.blackHoleSpin;
                    updated = true;
                    break;
            }
            if (updated) {
                this.updateUI();
            }
        });
        
        // Slider controls
        const spinSlider = document.getElementById('spinSlider');
        const innerRadiusSlider = document.getElementById('innerRadiusSlider');
        const outerRadiusSlider = document.getElementById('outerRadiusSlider');
        const volumetricToggle = document.getElementById('volumetricToggle');
        
        spinSlider.addEventListener('input', (e) => {
            this.blackHoleSpin = parseFloat(e.target.value);
            document.getElementById('spin').textContent = this.blackHoleSpin.toFixed(3);
        });
        
        innerRadiusSlider.addEventListener('input', (e) => {
            this.innerRadius = parseFloat(e.target.value);
            document.getElementById('innerRadius').textContent = this.innerRadius.toFixed(1);
        });
        
        outerRadiusSlider.addEventListener('input', (e) => {
            this.diskRadius = parseFloat(e.target.value);
            document.getElementById('radius').textContent = this.diskRadius.toFixed(1);
        });
        
        volumetricToggle.addEventListener('change', (e) => {
            this.volumetricMode = e.target.checked ? 1.0 : 0.0;
            document.getElementById('volumetricStatus').textContent = e.target.checked ? 'ON' : 'OFF';
        });

        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });
    }
    
    render() {
        this.updateCamera();
        this.updatePerformanceStats();
        
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
