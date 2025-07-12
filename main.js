class DiskVisualization {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.uniformBuffer = null;
        this.bindGroup = null;
        
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
            this.createRenderPipeline();
            this.createUniformBuffer();
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
                blackHoleMass: f32,
                blackHoleSpin: f32,
                screenWidth: f32,
                screenHeight: f32,
                observerDistance: f32,
                padding2: f32,
                viewMatrix: mat4x4<f32>,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            struct BoyerLindquistMetric {
                a: f32,
                M: f32,
                alpha: f32,
                beta3: f32,
                gamma11: f32,
                gamma22: f32,
                gamma33: f32,
                g_00: f32,
                g_03: f32,
                g_11: f32,
                g_22: f32,
                g_33: f32,
                d_alpha_dr: f32,
                d_beta3_dr: f32,
                d_gamma11_dr: f32,
                d_gamma22_dr: f32,
                d_gamma33_dr: f32,
                d_alpha_dth: f32,
                d_beta3_dth: f32,
                d_gamma11_dth: f32,
                d_gamma22_dth: f32,
                d_gamma33_dth: f32,
                delta: f32,
                sigma: f32,
                rho2: f32,
            }
            
            fn square(x: f32) -> f32 { return x * x; }
            fn cube(x: f32) -> f32 { return x * x * x; }
            
            fn computeMetric(r: f32, th: f32, a: f32, M: f32) -> BoyerLindquistMetric {
                var metric: BoyerLindquistMetric;
                metric.a = a;
                metric.M = M;
                
                let sth = sin(th);
                let cth = cos(th);
                let c2th = cos(2.0 * th);
                let s2th = sin(2.0 * th);
                let cscth = 1.0 / sth;
                let a2 = a * a;
                let a3 = a * a2;
                let a4 = a2 * a2;
                let a5 = a4 * a;
                let a6 = a2 * a4;
                let r2 = r * r;
                let r3 = r2 * r;
                let r4 = r2 * r2;
                let r6 = r2 * r4;
                
                metric.delta = r * r - 2.0 * M * r + a * a;
                metric.sigma = square(r2 + a2) - a2 * metric.delta * square(sth);
                metric.rho2 = r2 + a * a * cth * cth;
                
                metric.alpha = sqrt(metric.rho2 * metric.delta / metric.sigma);
                metric.beta3 = -2.0 * M * a * r / metric.sigma;
                
                metric.g_00 = 2.0 * M * r / metric.rho2 - 1.0;
                metric.g_03 = -2.0 * M * a * r / metric.rho2 * square(sth);
                metric.g_11 = metric.rho2 / metric.delta;
                metric.g_22 = metric.rho2;
                metric.g_33 = metric.sigma * square(sth) / metric.rho2;
                
                metric.gamma11 = metric.delta / metric.rho2;
                metric.gamma22 = 1.0 / metric.rho2;
                metric.gamma33 = metric.rho2 / metric.sigma / square(sth);
                
                // Derivatives
                metric.d_alpha_dr = M * 
                    (-a6 + 2.0 * r6 + a2 * r3 * (3.0 * r - 4.0 * M) -
                     a2 * (a4 + 2.0 * a2 * r2 + r3 * (r - 4.0 * M)) * c2th) /
                    (2.0 * metric.sigma * metric.sigma * sqrt(metric.delta * metric.rho2 / metric.sigma));
                
                metric.d_beta3_dr = M *
                    (-a5 + 3.0 * a3 * r2 + 6.0 * a * r4 + a3 * (r2 - a2) * c2th) /
                    square(metric.sigma);
                
                metric.d_gamma11_dr = 2.0 * (r * (M * r - a2) + a2 * (r - M) * square(cth)) /
                    square(metric.rho2);
                
                metric.d_gamma22_dr = -2.0 * r / square(metric.rho2);
                
                metric.d_gamma33_dr = 
                    (-2.0 * a4 * (r - M) * square(cscth) +
                     2.0 * (a2 * (2.0 * r - M) + r2 * (2.0 * r + M)) * square(a * square(cscth)) -
                     2.0 * r * square(a2 + r2) * square(cube(cscth))) /
                    square(a4 + a2 * r * (r - 2.0 * M) - square((a2 + r2) * cscth));
                
                metric.d_alpha_dth = -M * a2 * metric.delta * r * (a2 + r2) * s2th / square(metric.sigma) /
                    sqrt(metric.delta * metric.rho2 / metric.sigma);
                
                metric.d_beta3_dth = -2.0 * M * a3 * r * metric.delta * s2th / square(metric.sigma);
                
                metric.d_gamma11_dth = a2 * metric.delta * s2th / square(metric.rho2);
                metric.d_gamma22_dth = a2 * s2th / square(metric.rho2);
                
                metric.d_gamma33_dth = 
                    2.0 * (-a4 * metric.delta + 2.0 * a2 * metric.delta * (a2 + r2) * square(cscth) -
                           cube(a2 + r2) * square(square(cscth))) * cth / cube(sth) /
                    square(a4 + a2 * r * (r - 2.0 * M) - square((a2 + r2) * cscth));
                
                return metric;
            }
            
            fn u0(metric: BoyerLindquistMetric, u_1: f32, u_2: f32, u_3: f32) -> f32 {
                return sqrt(metric.gamma11 * u_1 * u_1 + metric.gamma22 * u_2 * u_2 + 
                           metric.gamma33 * u_3 * u_3) / metric.alpha;
            }
            
            struct GeodesicState {
                r: f32,
                theta: f32,
                phi: f32,
                ur: f32,
                utheta: f32,
                uphi: f32,
            }
            
            fn geodesicDerivatives(state: GeodesicState, a: f32, M: f32) -> GeodesicState {
                let metric = computeMetric(state.r, state.theta, a, M);
                let u_0 = u0(metric, state.ur, state.utheta, state.uphi);
                
                var derivs: GeodesicState;
                
                // Position derivatives
                derivs.r = metric.gamma11 * state.ur / u_0;
                derivs.theta = metric.gamma22 * state.utheta / u_0;
                derivs.phi = metric.gamma33 * state.uphi / u_0 - metric.beta3;
                
                // Momentum derivatives from geodesic equation
                derivs.ur = -metric.alpha * u_0 * metric.d_alpha_dr +
                           state.uphi * metric.d_beta3_dr -
                           (square(state.ur) * metric.d_gamma11_dr +
                            square(state.utheta) * metric.d_gamma22_dr +
                            square(state.uphi) * metric.d_gamma33_dr) / (2.0 * u_0);
                
                derivs.utheta = -metric.alpha * u_0 * metric.d_alpha_dth +
                               state.uphi * metric.d_beta3_dth -
                               (square(state.ur) * metric.d_gamma11_dth +
                                square(state.utheta) * metric.d_gamma22_dth +
                                square(state.uphi) * metric.d_gamma33_dth) / (2.0 * u_0);
                
                derivs.uphi = 0.0; // Conserved quantity
                
                return derivs;
            }
            
            fn traceGeodesic(rayOrigin: vec3<f32>, rayDir: vec3<f32>, a: f32, M: f32, diskRadius: f32, innerRadius: f32, maxDistance: f32) -> bool {
                let r0 = length(rayOrigin);
                let theta0 = acos(clamp(rayOrigin.z / r0, -1.0, 1.0));
                let phi0 = atan2(rayOrigin.y, rayOrigin.x);
                
                let rs = M + sqrt(M * M - a * a); // Event horizon
                if (r0 < rs * 1.01) {
                    return false;
                }
                
                // Initial conditions for geodesic
                var state: GeodesicState;
                state.r = r0;
                state.theta = theta0;
                state.phi = phi0;
                
                let metric = computeMetric(r0, theta0, a, M);
                
                // Better initial momentum setup - convert ray direction more carefully
                var sth0 = sin(theta0);
                if (sth0 < 1e-10) { sth0 = 1e-10; }
                
                // Project ray direction into spherical coordinates
                let r_hat = normalize(rayOrigin);
                let theta_hat = vec3<f32>(-r_hat.z * r_hat.x, -r_hat.z * r_hat.y, r_hat.x * r_hat.x + r_hat.y * r_hat.y) / sqrt(r_hat.x * r_hat.x + r_hat.y * r_hat.y + 1e-10);
                let phi_hat = vec3<f32>(-r_hat.y, r_hat.x, 0.0) / sqrt(r_hat.x * r_hat.x + r_hat.y * r_hat.y + 1e-10);
                
                let rdot = dot(rayDir, r_hat);
                let thetadot = dot(rayDir, theta_hat);
                let phidot = dot(rayDir, phi_hat);
                
                // Set momentum using proper normalization
                let normalization = sqrt(metric.g_11 * rdot * rdot + metric.g_22 * thetadot * thetadot + metric.g_33 * phidot * phidot);
                state.ur = rdot / normalization * sqrt(metric.g_11);
                state.utheta = thetadot / normalization * sqrt(metric.g_22);
                state.uphi = phidot / normalization * sqrt(metric.g_33);
                
                // RK45 Dormand-Prince integration with adaptive stepping
                var h = 0.1; // Initial step size
                let atol = 1e-6;
                let rtol = 1e-6;
                let hmin = 1e-4;
                let hmax = 2.0;
                let maxSteps = 10000;
                
                // RK45 Dormand-Prince coefficients
                let c2 = 1.0 / 5.0;
                let c3 = 3.0 / 10.0;
                let c4 = 4.0 / 5.0;
                let c5 = 8.0 / 9.0;
                
                let a21 = 1.0 / 5.0;
                let a31 = 3.0 / 40.0;
                let a32 = 9.0 / 40.0;
                let a41 = 44.0 / 45.0;
                let a42 = -56.0 / 15.0;
                let a43 = 32.0 / 9.0;
                let a51 = 19372.0 / 6561.0;
                let a52 = -25360.0 / 2187.0;
                let a53 = 64448.0 / 6561.0;
                let a54 = -212.0 / 729.0;
                let a61 = 9017.0 / 3168.0;
                let a62 = -355.0 / 33.0;
                let a63 = 46732.0 / 5247.0;
                let a64 = 49.0 / 176.0;
                let a65 = -5103.0 / 18656.0;
                
                let a71 = 35.0 / 384.0;
                let a73 = 500.0 / 1113.0;
                let a74 = 125.0 / 192.0;
                let a75 = -2187.0 / 6784.0;
                let a76 = 11.0 / 84.0;
                
                let e1 = 71.0 / 57600.0;
                let e3 = -71.0 / 16695.0;
                let e4 = 71.0 / 1920.0;
                let e5 = -17253.0 / 339200.0;
                let e6 = 22.0 / 525.0;
                let e7 = -1.0 / 40.0;
                
                var k1 = geodesicDerivatives(state, a, M);
                var err_prev = 1.0;
                
                for (var step = 0; step < maxSteps; step++) {
                    let prevZ = state.r * cos(state.theta);
                    
                    // RK45 step
                    var temp: GeodesicState;
                    temp.r = state.r + h * a21 * k1.r;
                    temp.theta = state.theta + h * a21 * k1.theta;
                    temp.phi = state.phi + h * a21 * k1.phi;
                    temp.ur = state.ur + h * a21 * k1.ur;
                    temp.utheta = state.utheta + h * a21 * k1.utheta;
                    temp.uphi = state.uphi + h * a21 * k1.uphi;
                    let k2 = geodesicDerivatives(temp, a, M);
                    
                    temp.r = state.r + h * (a31 * k1.r + a32 * k2.r);
                    temp.theta = state.theta + h * (a31 * k1.theta + a32 * k2.theta);
                    temp.phi = state.phi + h * (a31 * k1.phi + a32 * k2.phi);
                    temp.ur = state.ur + h * (a31 * k1.ur + a32 * k2.ur);
                    temp.utheta = state.utheta + h * (a31 * k1.utheta + a32 * k2.utheta);
                    temp.uphi = state.uphi + h * (a31 * k1.uphi + a32 * k2.uphi);
                    let k3 = geodesicDerivatives(temp, a, M);
                    
                    temp.r = state.r + h * (a41 * k1.r + a42 * k2.r + a43 * k3.r);
                    temp.theta = state.theta + h * (a41 * k1.theta + a42 * k2.theta + a43 * k3.theta);
                    temp.phi = state.phi + h * (a41 * k1.phi + a42 * k2.phi + a43 * k3.phi);
                    temp.ur = state.ur + h * (a41 * k1.ur + a42 * k2.ur + a43 * k3.ur);
                    temp.utheta = state.utheta + h * (a41 * k1.utheta + a42 * k2.utheta + a43 * k3.utheta);
                    temp.uphi = state.uphi + h * (a41 * k1.uphi + a42 * k2.uphi + a43 * k3.uphi);
                    let k4 = geodesicDerivatives(temp, a, M);
                    
                    temp.r = state.r + h * (a51 * k1.r + a52 * k2.r + a53 * k3.r + a54 * k4.r);
                    temp.theta = state.theta + h * (a51 * k1.theta + a52 * k2.theta + a53 * k3.theta + a54 * k4.theta);
                    temp.phi = state.phi + h * (a51 * k1.phi + a52 * k2.phi + a53 * k3.phi + a54 * k4.phi);
                    temp.ur = state.ur + h * (a51 * k1.ur + a52 * k2.ur + a53 * k3.ur + a54 * k4.ur);
                    temp.utheta = state.utheta + h * (a51 * k1.utheta + a52 * k2.utheta + a53 * k3.utheta + a54 * k4.utheta);
                    temp.uphi = state.uphi + h * (a51 * k1.uphi + a52 * k2.uphi + a53 * k3.uphi + a54 * k4.uphi);
                    let k5 = geodesicDerivatives(temp, a, M);
                    
                    temp.r = state.r + h * (a61 * k1.r + a62 * k2.r + a63 * k3.r + a64 * k4.r + a65 * k5.r);
                    temp.theta = state.theta + h * (a61 * k1.theta + a62 * k2.theta + a63 * k3.theta + a64 * k4.theta + a65 * k5.theta);
                    temp.phi = state.phi + h * (a61 * k1.phi + a62 * k2.phi + a63 * k3.phi + a64 * k4.phi + a65 * k5.phi);
                    temp.ur = state.ur + h * (a61 * k1.ur + a62 * k2.ur + a63 * k3.ur + a64 * k4.ur + a65 * k5.ur);
                    temp.utheta = state.utheta + h * (a61 * k1.utheta + a62 * k2.utheta + a63 * k3.utheta + a64 * k4.utheta + a65 * k5.utheta);
                    temp.uphi = state.uphi + h * (a61 * k1.uphi + a62 * k2.uphi + a63 * k3.uphi + a64 * k4.uphi + a65 * k5.uphi);
                    let k6 = geodesicDerivatives(temp, a, M);
                    
                    var y_next: GeodesicState;
                    y_next.r = state.r + h * (a71 * k1.r + a73 * k3.r + a74 * k4.r + a75 * k5.r + a76 * k6.r);
                    y_next.theta = state.theta + h * (a71 * k1.theta + a73 * k3.theta + a74 * k4.theta + a75 * k5.theta + a76 * k6.theta);
                    y_next.phi = state.phi + h * (a71 * k1.phi + a73 * k3.phi + a74 * k4.phi + a75 * k5.phi + a76 * k6.phi);
                    y_next.ur = state.ur + h * (a71 * k1.ur + a73 * k3.ur + a74 * k4.ur + a75 * k5.ur + a76 * k6.ur);
                    y_next.utheta = state.utheta + h * (a71 * k1.utheta + a73 * k3.utheta + a74 * k4.utheta + a75 * k5.utheta + a76 * k6.utheta);
                    y_next.uphi = state.uphi + h * (a71 * k1.uphi + a73 * k3.uphi + a74 * k4.uphi + a75 * k5.uphi + a76 * k6.uphi);
                    let k7 = geodesicDerivatives(y_next, a, M);
                    
                    // Error estimation
                    let y_err_r = h * (e1 * k1.r + e3 * k3.r + e4 * k4.r + e5 * k5.r + e6 * k6.r + e7 * k7.r);
                    let y_err_theta = h * (e1 * k1.theta + e3 * k3.theta + e4 * k4.theta + e5 * k5.theta + e6 * k6.theta + e7 * k7.theta);
                    let y_err_phi = h * (e1 * k1.phi + e3 * k3.phi + e4 * k4.phi + e5 * k5.phi + e6 * k6.phi + e7 * k7.phi);
                    let y_err_ur = h * (e1 * k1.ur + e3 * k3.ur + e4 * k4.ur + e5 * k5.ur + e6 * k6.ur + e7 * k7.ur);
                    let y_err_utheta = h * (e1 * k1.utheta + e3 * k3.utheta + e4 * k4.utheta + e5 * k5.utheta + e6 * k6.utheta + e7 * k7.utheta);
                    let y_err_uphi = h * (e1 * k1.uphi + e3 * k3.uphi + e4 * k4.uphi + e5 * k5.uphi + e6 * k6.uphi + e7 * k7.uphi);
                    
                    // Compute error norm
                    var err = 0.0;
                    err += square(y_err_r / (atol + max(abs(state.r), abs(y_next.r)) * rtol));
                    err += square(y_err_theta / (atol + max(abs(state.theta), abs(y_next.theta)) * rtol));
                    err += square(y_err_phi / (atol + max(abs(state.phi), abs(y_next.phi)) * rtol));
                    err += square(y_err_ur / (atol + max(abs(state.ur), abs(y_next.ur)) * rtol));
                    err += square(y_err_utheta / (atol + max(abs(state.utheta), abs(y_next.utheta)) * rtol));
                    err += square(y_err_uphi / (atol + max(abs(state.uphi), abs(y_next.uphi)) * rtol));
                    err = sqrt(err / 6.0);
                    err = max(err, 1e-10);
                    
                    // Accept or reject step
                    if (err < 1.0) {
                        // Accept step
                        state = y_next;
                        k1 = k7;
                        err_prev = err;
                        
                        // Check stopping conditions
                        if (state.r < rs * 1.01) {
                            return false; // Hit event horizon
                        }
                        
                        if (state.r > maxDistance) {
                            return false; // Escaped to infinity
                        }
                        
                        let currentZ = state.r * cos(state.theta);
                        
                        // Check for disk crossing
                        if (prevZ * currentZ < 0.0 && state.r >= innerRadius && state.r <= diskRadius) {
                            return true;
                        }
                    }
                    
                    // Adaptive step size control
                    let S = 0.9; // Safety factor
                    if (err_prev < 1.0) {
                        // Previous step was accepted
                        let err_alpha = 0.7 / 5.0;
                        let err_beta = 0.4 / 5.0;
                        h = S * h * pow(err, -err_alpha) * pow(err_prev, err_beta);
                    } else {
                        // Previous step was rejected
                        h = min(h, S * h * pow(1.0 / err, 0.2));
                    }
                    h = max(h, hmin);
                    h = min(h, hmax);
                }
                
                return false;
            }
            
            @fragment
            fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
                // Block-based rendering - compute only every 8th pixel, use for entire 8x8 block
                let blockSize = 8;
                let pixelX = i32(fragCoord.x);
                let pixelY = i32(fragCoord.y);
                let blockX = (pixelX / blockSize) * blockSize;
                let blockY = (pixelY / blockSize) * blockSize;
                
                // Use the center of the block for computation
                let centerX = f32(blockX + blockSize / 2);
                let centerY = f32(blockY + blockSize / 2);
                
                let blockScreenPos = vec2<f32>(
                    (centerX / uniforms.screenWidth) * 2.0 - 1.0,
                    1.0 - (centerY / uniforms.screenHeight) * 2.0
                );
                
                // Use block center for ray computation
                let aspectRatio = uniforms.screenWidth / uniforms.screenHeight;
                let ndc = vec2<f32>(blockScreenPos.x * aspectRatio, blockScreenPos.y);
                
                let fov = 0.785398;
                let rayDirLocal = normalize(vec3<f32>(ndc.x * tan(fov/2.0), ndc.y * tan(fov/2.0), -1.0));
                
                let viewMatrixInv = transpose(uniforms.viewMatrix);
                let rayDir = normalize((viewMatrixInv * vec4<f32>(rayDirLocal, 0.0)).xyz);
                
                // Test center region with GR, outer with classical (use original pixel for region test)
                let screenPos = vec2<f32>(
                    (fragCoord.x / uniforms.screenWidth) * 2.0 - 1.0,
                    1.0 - (fragCoord.y / uniforms.screenHeight) * 2.0
                );
                
                if (length(blockScreenPos) < 0.3) {
                    // Center region: test GR
                    let grHit = traceGeodesic(uniforms.cameraPos, rayDir, uniforms.blackHoleSpin, uniforms.blackHoleMass, uniforms.diskRadius, uniforms.innerRadius, uniforms.observerDistance * 2.0);
                    if (grHit) {
                        return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta for GR hit
                    } else {
                        return vec4<f32>(0.0, 0.0, 1.0, 1.0); // Blue for GR miss
                    }
                } else {
                    // Outer region: classical ray tracing
                    if (abs(rayDir.z) > 0.0001) {
                        let t = -uniforms.cameraPos.z / rayDir.z;
                        if (t > 0.0) {
                            let hitPoint = uniforms.cameraPos + t * rayDir;
                            let distFromCenter = length(hitPoint.xy);
                            
                            if (distFromCenter <= uniforms.diskRadius && distFromCenter >= uniforms.innerRadius) {
                                return vec4<f32>(0.0, 1.0, 0.0, 1.0); // Green for classical hit
                            }
                        }
                    }
                }
                
                return vec4<f32>(0.0, 0.0, 0.0, 1.0); // Black background
            }
        `;
        
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
        const uniformBufferSize = 4 * 4 + 4 * 8 + 4 * 16;
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
        
        const uniformData = new Float32Array(4 + 8 + 16);
        
        uniformData[0] = x;
        uniformData[1] = y;
        uniformData[2] = z;
        uniformData[3] = 0;
        
        uniformData[4] = this.diskRadius;
        uniformData[5] = this.innerRadius;
        uniformData[6] = this.blackHoleMass;
        uniformData[7] = this.blackHoleSpin;
        uniformData[8] = this.canvas.width;
        uniformData[9] = this.canvas.height;
        uniformData[10] = this.observerDistance;
        uniformData[11] = 0;
        
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
