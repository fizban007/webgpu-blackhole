struct Uniforms {
    cameraPos: vec3<f32>,
    time: f32,
    diskRadius: f32,
    innerRadius: f32,
    blackHoleMass: f32,
    blackHoleSpin: f32,
    screenWidth: f32,
    screenHeight: f32,
    observerDistance: f32,
    volumetricMode: f32, // 0.0 = thin disk, 1.0 = volumetric
    viewMatrix: mat4x4<f32>,
    starPosition: vec3<f32>,
    starRadius: f32,
    starColor: vec3<f32>,
    starPadding: f32, // Padding for alignment
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

// Simplex noise functions for disk texture
fn mod289(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_2(x: vec2<f32>) -> vec2<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec3<f32>) -> vec3<f32> {
    return mod289(((x * 34.0) + 10.0) * x);
}

fn simplexNoise2D(v: vec2<f32>) -> f32 {
    let C = vec4<f32>(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    
    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);
    
    var i1: vec2<f32>;
    if (x0.x > x0.y) {
        i1 = vec2<f32>(1.0, 0.0);
    } else {
        i1 = vec2<f32>(0.0, 1.0);
    }
    
    var x12 = x0.xyxy + C.xxzz;
    x12.x = x12.x - i1.x;
    x12.y = x12.y - i1.y;
    
    i = mod289_2(i);
    let p = permute(permute(i.y + vec3<f32>(0.0, i1.y, 1.0)) + i.x + vec3<f32>(0.0, i1.x, 1.0));
    
    var m = max(vec3<f32>(0.5) - vec3<f32>(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3<f32>(0.0));
    m = m * m;
    m = m * m;
    
    let x = 2.0 * fract(p * C.www) - 1.0;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;
    
    m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));
    
    let g = vec3<f32>(
        a0.x * x0.x + h.x * x0.y,
        a0.y * x12.x + h.y * x12.y,
        a0.z * x12.z + h.z * x12.w
    );
    
    return 130.0 * dot(m, g);
}

// Fractal noise for more detail
fn fractalNoise(p: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var maxValue = 0.0;
    
    for (var i = 0; i < octaves; i++) {
        value += simplexNoise2D(p * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value / maxValue;
}

// Periodic noise using sine/cosine for seamless wrapping
fn periodicNoise2D(p: vec2<f32>, period: vec2<f32>) -> f32 {
    // Convert to periodic coordinates
    let theta = p * 6.28318530718 / period;
    
    // 4D coordinates for seamless tiling
    let nx = cos(theta.x);
    let ny = sin(theta.x);
    let nz = cos(theta.y);
    let nw = sin(theta.y);
    
    // Sample 4D noise at these coordinates
    let a = simplexNoise2D(vec2<f32>(nx, nz));
    let b = simplexNoise2D(vec2<f32>(ny, nw));
    let c = simplexNoise2D(vec2<f32>(nx + ny, nz + nw));
    
    return (a + b + c) / 3.0;
}

// Fractal periodic noise
fn fractalPeriodicNoise(p: vec2<f32>, period: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var maxValue = 0.0;
    
    for (var i = 0; i < octaves; i++) {
        value += periodicNoise2D(p * frequency, period * frequency) * amplitude;
        maxValue += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value / maxValue;
}

fn boyerLindquistToCartesian(r: f32, theta: f32, phi: f32) -> vec3<f32> {
    let x = r * sin(theta) * cos(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(theta);
    return vec3<f32>(x, y, z);
}

struct StarSample {
    density: f32,
    emission: vec3<f32>,
}

fn getStarDensity(r: f32, theta: f32, phi: f32) -> StarSample {
    var sample: StarSample;
    sample.density = 0.0;
    sample.emission = vec3<f32>(0.0);
    
    let rayPos = boyerLindquistToCartesian(r, theta, phi);
    let distance = length(rayPos - uniforms.starPosition);
    
    if (distance > uniforms.starRadius) {
        return sample;
    }
    
    // Normalized distance from star center (0 at center, 1 at surface)
    let normalizedDist = distance / uniforms.starRadius;
    
    // Quick falloff density profile - high density at center, drops off quickly
    // Using exponential falloff for fast opacity buildup
    let densityFalloff = exp(-normalizedDist / 3.0); // Steep falloff
    sample.density = densityFalloff * 1.0; // High density multiplier for quick opacity
    
    // Star emission with temperature-based color variation
    let temperature = 1.0 - normalizedDist * 0.1; // Hotter at center
    let baseIntensity = densityFalloff * temperature * 2.0;
    
    // Color varies from reddish at edges to white-hot at center
    let hotColor = vec3<f32>(1.0, 1.0, 1.0); // White hot
    let coolColor = uniforms.starColor; // Reddish base color
    
    // Blend between cool and hot based on density (higher density = hotter = whiter)
    let colorTemperature = densityFalloff; // Use density as temperature indicator
    let finalColor = mix(coolColor, hotColor, colorTemperature * temperature);
    
    sample.emission = finalColor * baseIntensity;
    
    return sample;
}

fn computeMetric(r: f32, th: f32, a1: f32, M: f32) -> BoyerLindquistMetric {
  var metric: BoyerLindquistMetric;
  let a = a1 * M;
  metric.a = a;
  metric.M = M;
    
  let sth = sin(th);
  let cth = cos(th);
  let sth2 = sth * sth;
  let cth2 = cth * cth;
  let c2th = 2.0 * cth2 - 1.0; // cos(2θ) = 2cos²θ - 1
  let s2th = 2.0 * sth * cth;  // sin(2θ) = 2sinθcosθ
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
    
  metric.delta = r2 - 2.0 * M * r + a2;
  metric.sigma = square(r2 + a2) - a2 * metric.delta * sth2;
  metric.rho2 = r2 + a2 * cth2;
    
  metric.alpha = sqrt(metric.rho2 * metric.delta / metric.sigma);
  metric.beta3 = -2.0 * M * a * r / metric.sigma;
    
  metric.g_00 = 2.0 * M * r / metric.rho2 - 1.0;
  metric.g_03 = -2.0 * M * a * r / metric.rho2 * sth2;
  metric.g_11 = metric.rho2 / metric.delta;
  metric.g_22 = metric.rho2;
  metric.g_33 = metric.sigma * sth2 / metric.rho2;
    
  metric.gamma11 = metric.delta / metric.rho2;
  metric.gamma22 = 1.0 / metric.rho2;
  metric.gamma33 = metric.rho2 / metric.sigma / sth2;
    
  // Derivatives
  metric.d_alpha_dr = M *
      (-a6 + 2.0 * r6 + a2 * r3 * (3.0 * r - 4.0 * M) -
       a2 * (a4 + 2.0 * a2 * r2 + r3 * (r - 4.0 * M)) * c2th) /
      (2.0 * metric.sigma * metric.sigma * sqrt(metric.delta * metric.rho2 / metric.sigma));
    
  metric.d_beta3_dr = M *
      (-a5 + 3.0 * a3 * r2 + 6.0 * a * r4 + a3 * (r2 - a2) * c2th) /
      square(metric.sigma);
    
  metric.d_gamma11_dr = 2.0 * (r * (M * r - a2) + a2 * (r - M) * cth2) /
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

struct DiskHit {
    hit: bool,
    r: f32,
    phi: f32,
}

struct VolumetricSample {
    density: f32,
    emission: vec3<f32>,
}

fn geodesicDerivatives(state: GeodesicState, a: f32, M: f32) -> GeodesicState {
  let metric = computeMetric(state.r, state.theta, a, M);
  let u_0 = u0(metric, state.ur, state.utheta, state.uphi);
  let inv_u0 = 1.0 / u_0;
  let inv_2u0 = 1.0 / (2.0 * u_0);
    
  var derivs: GeodesicState;
    
  // Position derivatives - precompute inverse
  derivs.r = metric.gamma11 * state.ur * inv_u0;
  derivs.theta = metric.gamma22 * state.utheta * inv_u0;
  derivs.phi = metric.gamma33 * state.uphi * inv_u0 - metric.beta3;
    
  // Precompute squared momentum terms
  let ur2 = state.ur * state.ur;
  let utheta2 = state.utheta * state.utheta;
  let uphi2 = state.uphi * state.uphi;
    
  // Momentum derivatives from geodesic equation
  derivs.ur = -metric.alpha * u_0 * metric.d_alpha_dr +
      state.uphi * metric.d_beta3_dr -
      (ur2 * metric.d_gamma11_dr +
       utheta2 * metric.d_gamma22_dr +
       uphi2 * metric.d_gamma33_dr) * inv_2u0;
    
  derivs.utheta = -metric.alpha * u_0 * metric.d_alpha_dth +
      state.uphi * metric.d_beta3_dth -
      (ur2 * metric.d_gamma11_dth +
       utheta2 * metric.d_gamma22_dth +
       uphi2 * metric.d_gamma33_dth) * inv_2u0;
    
  derivs.uphi = 0.0; // Conserved quantity
    
  return derivs;
}

fn getDiskDensity(r: f32, z: f32, phi: f32, innerRadius: f32, outerRadius: f32) -> VolumetricSample {
    var sample: VolumetricSample;
    sample.density = 0.0;
    sample.emission = vec3<f32>(0.0);
    
    // Disk height profile - exponential falloff from midplane
    let diskHeight = 0.5 + (r - innerRadius) / (outerRadius - innerRadius) * 2.0; // Height increases with radius
    // let heightFalloff = exp(-abs(z) / diskHeight);
    let heightFalloff = 1.0 / (square(abs(z) / diskHeight) + 1.0);
    
    // Radial density profile
    let radialNorm = (r - innerRadius) / (outerRadius - innerRadius);
    if (radialNorm < 0.0 || radialNorm > 1.0) {
        return sample;
    }
    
    // Power law density profile with inner edge enhancement
    let radialDensity = pow(1.0 - radialNorm, 0.5) * (1.0 + 3.0 * exp(-radialNorm * 5.0));
    
    // Add time-varying rotation and fluctuation
    let timeOffset = uniforms.time * 0.1; // Slow rotation
    // let fluctuation = sin(-uniforms.time * 0.3) * 0.5 + cos(uniforms.time * 0.8) * 0.4 + 0.5; // Temporal fluctuation
    let fluctuation = sin(-uniforms.time * 0.3) * 0.5 + 0.5; // Temporal fluctuation
    
    // Add volumetric noise for cloud structure with time animation
    // Use periodic coordinates for seamless wrapping in phi
    let phiNorm = (phi + 3.14159265359) / 6.28318530718; // Normalize phi to [0,1]
    let animatedPhi = phi - timeOffset * 2.0; // Rotate over time
    
    // Create periodic noise coordinates using sin/cos for seamless wrapping
    let phiCos = cos(animatedPhi);
    let phiSin = sin(animatedPhi);
    
    // Time-varying noise with fluctuation
    let noise1 = simplexNoise2D(vec2<f32>(r * 0.1, z * 2.0 + uniforms.time * 0.05)) * 0.5 + 0.5;
    let noise2 = simplexNoise2D(vec2<f32>(phiCos * 3.0 + r * 0.05, phiSin * 3.0 + z * 0.5 + uniforms.time * 0.02)) * 0.5 + 0.5;
    
    // Animated periodic noise for phi variations
    let animatedPhiNorm = fract(phiNorm + timeOffset * 0.5);
    let periodicNoise1 = periodicNoise2D(vec2<f32>(r * 0.2, animatedPhiNorm * 4.0 + uniforms.time * 0.1), vec2<f32>(10.0, 1.0));
    
    // Simplified turbulence calculation with time variation
    let turbulence = (noise1 * 0.4 + noise2 * 0.4 + (periodicNoise1 * 0.5 + 0.5) * 0.2) * (0.8 + fluctuation * 0.4);
    
    // Spiral arm structure with rotation
    let spiralAngle = animatedPhi + r * 0.2;
    let spiralPattern = sin(spiralAngle * 3.0) * 0.5 + 0.5;
    let spiralDensity = mix(0.3, 1.0, spiralPattern * turbulence);
    
    // Final density
    sample.density = radialDensity * heightFalloff * spiralDensity * 0.8;
    
    // Temperature and emission based on radius
    let temperature = 1.0 - radialNorm * 0.7; // Hotter near inner edge
    let emission_intensity = temperature * radialDensity * 3.0;
    
    // Color based on temperature - blue-white hot to red cool
    let hotColor = vec3<f32>(0.7, 0.8, 1.0); // Blue-white
    let warmColor = vec3<f32>(1.0, 0.5, 0.2); // Orange
    let coolColor = vec3<f32>(0.8, 0.1, 0.05); // Red
    
    let color = mix(mix(coolColor, warmColor, temperature), hotColor, pow(temperature, 2.0));
    sample.emission = color * emission_intensity * (0.5 + turbulence * 0.5);
    
    return sample;
}

struct RK45State {
  h: f32,
  err_prev: f32,
  k1: GeodesicState,
}

fn performRK4Step(state: ptr<function, GeodesicState>, h: f32, a: f32, M: f32) {
  // Simple RK4 step with fixed step size for use near poles
  let k1 = geodesicDerivatives(*state, a, M);
  
  var temp: GeodesicState;
  temp.r = (*state).r + h * 0.5 * k1.r;
  temp.theta = (*state).theta + h * 0.5 * k1.theta;
  temp.phi = (*state).phi + h * 0.5 * k1.phi;
  temp.ur = (*state).ur + h * 0.5 * k1.ur;
  temp.utheta = (*state).utheta + h * 0.5 * k1.utheta;
  temp.uphi = (*state).uphi + h * 0.5 * k1.uphi;
  let k2 = geodesicDerivatives(temp, a, M);
  
  temp.r = (*state).r + h * 0.5 * k2.r;
  temp.theta = (*state).theta + h * 0.5 * k2.theta;
  temp.phi = (*state).phi + h * 0.5 * k2.phi;
  temp.ur = (*state).ur + h * 0.5 * k2.ur;
  temp.utheta = (*state).utheta + h * 0.5 * k2.utheta;
  temp.uphi = (*state).uphi + h * 0.5 * k2.uphi;
  let k3 = geodesicDerivatives(temp, a, M);
  
  temp.r = (*state).r + h * k3.r;
  temp.theta = (*state).theta + h * k3.theta;
  temp.phi = (*state).phi + h * k3.phi;
  temp.ur = (*state).ur + h * k3.ur;
  temp.utheta = (*state).utheta + h * k3.utheta;
  temp.uphi = (*state).uphi + h * k3.uphi;
  let k4 = geodesicDerivatives(temp, a, M);
  
  // Update state with RK4 formula
  (*state).r += h * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r) / 6.0;
  (*state).theta += h * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta) / 6.0;
  (*state).phi += h * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi) / 6.0;
  (*state).ur += h * (k1.ur + 2.0 * k2.ur + 2.0 * k3.ur + k4.ur) / 6.0;
  (*state).utheta += h * (k1.utheta + 2.0 * k2.utheta + 2.0 * k3.utheta + k4.utheta) / 6.0;
  (*state).uphi += h * (k1.uphi + 2.0 * k2.uphi + 2.0 * k3.uphi + k4.uphi) / 6.0;
}

fn performRK45Step(state: ptr<function, GeodesicState>, rk45_state: ptr<function, RK45State>, a: f32, M: f32, atol: f32, rtol: f32, hmin: f32, hmax: f32) -> bool {
  // Precomputed RK45 Dormand-Prince coefficients
  const a21 = 0.2;
  const a31 = 0.075;
  const a32 = 0.225;
  const a41 = 0.977777777778;
  const a42 = -3.733333333333;
  const a43 = 3.555555555556;
  const a51 = 2.952598689758;
  const a52 = -11.595793324188;
  const a53 = 9.822892851699;
  const a54 = -0.290793779983;
  const a61 = 2.846275252525;
  const a62 = -10.757575757576;
  const a63 = 8.906422717744;
  const a64 = 0.278267045455;
  const a65 = -0.273459052841;
  const a71 = 0.091145833333;
  const a73 = 0.449236298936;
  const a74 = 0.651041666667;
  const a75 = -0.322376179245;
  const a76 = 0.130952380952;
  const e1 = 0.001234567901;
  const e3 = -0.004259930906;
  const e4 = 0.036979166667;
  const e5 = -0.050867449137;
  const e6 = 0.041904761905;
  const e7 = -0.025;

  let h = (*rk45_state).h;
  let k1 = (*rk45_state).k1;
  
  // RK45 step
  var temp: GeodesicState;
  temp.r = (*state).r + h * a21 * k1.r;
  temp.theta = (*state).theta + h * a21 * k1.theta;
  temp.phi = (*state).phi + h * a21 * k1.phi;
  temp.ur = (*state).ur + h * a21 * k1.ur;
  temp.utheta = (*state).utheta + h * a21 * k1.utheta;
  temp.uphi = (*state).uphi + h * a21 * k1.uphi;
  let k2 = geodesicDerivatives(temp, a, M);
  
  temp.r = (*state).r + h * (a31 * k1.r + a32 * k2.r);
  temp.theta = (*state).theta + h * (a31 * k1.theta + a32 * k2.theta);
  temp.phi = (*state).phi + h * (a31 * k1.phi + a32 * k2.phi);
  temp.ur = (*state).ur + h * (a31 * k1.ur + a32 * k2.ur);
  temp.utheta = (*state).utheta + h * (a31 * k1.utheta + a32 * k2.utheta);
  temp.uphi = (*state).uphi + h * (a31 * k1.uphi + a32 * k2.uphi);
  let k3 = geodesicDerivatives(temp, a, M);
  
  temp.r = (*state).r + h * (a41 * k1.r + a42 * k2.r + a43 * k3.r);
  temp.theta = (*state).theta + h * (a41 * k1.theta + a42 * k2.theta + a43 * k3.theta);
  temp.phi = (*state).phi + h * (a41 * k1.phi + a42 * k2.phi + a43 * k3.phi);
  temp.ur = (*state).ur + h * (a41 * k1.ur + a42 * k2.ur + a43 * k3.ur);
  temp.utheta = (*state).utheta + h * (a41 * k1.utheta + a42 * k2.utheta + a43 * k3.utheta);
  temp.uphi = (*state).uphi + h * (a41 * k1.uphi + a42 * k2.uphi + a43 * k3.uphi);
  let k4 = geodesicDerivatives(temp, a, M);
  
  temp.r = (*state).r + h * (a51 * k1.r + a52 * k2.r + a53 * k3.r + a54 * k4.r);
  temp.theta = (*state).theta + h * (a51 * k1.theta + a52 * k2.theta + a53 * k3.theta + a54 * k4.theta);
  temp.phi = (*state).phi + h * (a51 * k1.phi + a52 * k2.phi + a53 * k3.phi + a54 * k4.phi);
  temp.ur = (*state).ur + h * (a51 * k1.ur + a52 * k2.ur + a53 * k3.ur + a54 * k4.ur);
  temp.utheta = (*state).utheta + h * (a51 * k1.utheta + a52 * k2.utheta + a53 * k3.utheta + a54 * k4.utheta);
  temp.uphi = (*state).uphi + h * (a51 * k1.uphi + a52 * k2.uphi + a53 * k3.uphi + a54 * k4.uphi);
  let k5 = geodesicDerivatives(temp, a, M);
  
  temp.r = (*state).r + h * (a61 * k1.r + a62 * k2.r + a63 * k3.r + a64 * k4.r + a65 * k5.r);
  temp.theta = (*state).theta + h * (a61 * k1.theta + a62 * k2.theta + a63 * k3.theta + a64 * k4.theta + a65 * k5.theta);
  temp.phi = (*state).phi + h * (a61 * k1.phi + a62 * k2.phi + a63 * k3.phi + a64 * k4.phi + a65 * k5.phi);
  temp.ur = (*state).ur + h * (a61 * k1.ur + a62 * k2.ur + a63 * k3.ur + a64 * k4.ur + a65 * k5.ur);
  temp.utheta = (*state).utheta + h * (a61 * k1.utheta + a62 * k2.utheta + a63 * k3.utheta + a64 * k4.utheta + a65 * k5.utheta);
  temp.uphi = (*state).uphi + h * (a61 * k1.uphi + a62 * k2.uphi + a63 * k3.uphi + a64 * k4.uphi + a65 * k5.uphi);
  let k6 = geodesicDerivatives(temp, a, M);
  
  var y_next: GeodesicState;
  y_next.r = (*state).r + h * (a71 * k1.r + a73 * k3.r + a74 * k4.r + a75 * k5.r + a76 * k6.r);
  y_next.theta = (*state).theta + h * (a71 * k1.theta + a73 * k3.theta + a74 * k4.theta + a75 * k5.theta + a76 * k6.theta);
  y_next.phi = (*state).phi + h * (a71 * k1.phi + a73 * k3.phi + a74 * k4.phi + a75 * k5.phi + a76 * k6.phi);
  y_next.ur = (*state).ur + h * (a71 * k1.ur + a73 * k3.ur + a74 * k4.ur + a75 * k5.ur + a76 * k6.ur);
  y_next.utheta = (*state).utheta + h * (a71 * k1.utheta + a73 * k3.utheta + a74 * k4.utheta + a75 * k5.utheta + a76 * k6.utheta);
  y_next.uphi = (*state).uphi + h * (a71 * k1.uphi + a73 * k3.uphi + a74 * k4.uphi + a75 * k5.uphi + a76 * k6.uphi);
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
  err += square(y_err_r / (atol + max(abs((*state).r), abs(y_next.r)) * rtol));
  err += square(y_err_theta / (atol + max(abs((*state).theta), abs(y_next.theta)) * rtol));
  err += square(y_err_phi / (atol + max(abs((*state).phi), abs(y_next.phi)) * rtol));
  err += square(y_err_ur / (atol + max(abs((*state).ur), abs(y_next.ur)) * rtol));
  err += square(y_err_utheta / (atol + max(abs((*state).utheta), abs(y_next.utheta)) * rtol));
  err += square(y_err_uphi / (atol + max(abs((*state).uphi), abs(y_next.uphi)) * rtol));
  err = sqrt(err / 6.0);
  err = max(err, 1e-10);
  
  // Accept or reject step
  if (err < 1.0) {
    // Accept step
    *state = y_next;
    (*rk45_state).k1 = k7;
    (*rk45_state).err_prev = err;
    
    // Adaptive step size control
    let S = 0.9;
    if ((*rk45_state).err_prev < 1.0) {
      let err_alpha = 0.7 / 5.0;
      let err_beta = 0.4 / 5.0;
      (*rk45_state).h = S * h * pow(err, -err_alpha) * pow((*rk45_state).err_prev, err_beta);
    } else {
      (*rk45_state).h = min(h, S * h * pow(1.0 / err, 0.2));
    }
    (*rk45_state).h = max((*rk45_state).h, hmin);
    (*rk45_state).h = min((*rk45_state).h, hmax);
    
    return true; // Step accepted
  } else {
    // Reject step and adjust step size
    let S = 0.9;
    (*rk45_state).h = min(h, S * h * pow(1.0 / err, 0.2));
    (*rk45_state).h = max((*rk45_state).h, hmin);
    (*rk45_state).h = min((*rk45_state).h, hmax);
    
    return false; // Step rejected
  }
}

fn renderThinDisk(cylindricalRadius: f32, phi: f32, innerRadius: f32, diskRadius: f32) -> vec4<f32> {
  // Add time-varying rotation and fluctuation
  let timeOffset = uniforms.time * 0.08; // Disk rotation speed
  let fluctuation = sin(uniforms.time * 0.25) * 0.5 + 0.5; // Temporal fluctuation
  
  let animatedPhi = phi - timeOffset * 2.0;
  let phiNormalized = (animatedPhi + 3.14159265359) / 6.28318530718;
  let texCoord = vec2<f32>(cylindricalRadius * 0.2, phiNormalized);
  
  let period1 = vec2<f32>(20.0, 1.0);
  let period2 = vec2<f32>(8.0, 1.0);
  let period3 = vec2<f32>(3.0, 1.0);
  
  // Add time animation to noise coordinates
  let timeCoord1 = texCoord * 2.0 + vec2<f32>(uniforms.time * 0.03, 0.0);
  let timeCoord2 = texCoord * 5.0 + vec2<f32>(uniforms.time * 0.05, 0.0);
  let timeCoord3 = texCoord * 15.0 + vec2<f32>(uniforms.time * 0.02, 0.0);
  let timeCoord4 = texCoord * 40.0 + vec2<f32>(uniforms.time * 0.08, 0.0);
  
  let noise1 = fractalPeriodicNoise(timeCoord1, period1, 4);
  let noise2 = fractalPeriodicNoise(timeCoord2, period2, 3);
  let noise3 = fractalPeriodicNoise(timeCoord3, period3, 2);
  let noise4 = periodicNoise2D(timeCoord4, vec2<f32>(1.5, 1.0));
  
  let spiralCoord = vec2<f32>(cylindricalRadius * 0.15, phiNormalized * 3.0 + cylindricalRadius * 0.1 + uniforms.time * 0.04);
  let spiralNoise = fractalPeriodicNoise(spiralCoord, vec2<f32>(10.0, 1.0), 3);
  
  let radialTurbulence = fractalNoise(vec2<f32>(cylindricalRadius * 0.3, phiNormalized * 10.0 + uniforms.time * 0.06), 4);
  
  // Apply fluctuation to turbulence
  let turbulence = (noise1 * 0.3 + noise2 * 0.25 + noise3 * 0.2 + noise4 * 0.15 + spiralNoise * 0.05 + radialTurbulence * 0.05) * (0.7 + fluctuation * 0.6);
  
  let heat = 0.7 + turbulence * 0.3;
  let baseColor = vec3<f32>(heat * 1.2, heat * 0.2, heat * 0.1);
  
  let hotSpots1 = max(0.0, noise1 - 0.4) * 1.5;
  let hotSpots2 = max(0.0, noise2 - 0.5) * 1.0;
  let hotSpots3 = max(0.0, spiralNoise - 0.3) * 0.8;
  let totalHotSpots = hotSpots1 + hotSpots2 + hotSpots3;
  
  let color = baseColor + vec3<f32>(totalHotSpots, totalHotSpots * 0.7, totalHotSpots * 0.2);
  
  let radialFactor = 1.0 - pow((cylindricalRadius - innerRadius) / (diskRadius - innerRadius), 2.0);
  let finalColor = color * (0.5 + radialFactor * 0.5);
  
  return vec4<f32>(finalColor, 1.0);
}

fn traceGeodesicThinDisk(rayOrigin: vec3<f32>, rayDir: vec3<f32>, a: f32, M: f32, diskRadius: f32, innerRadius: f32, maxDistance: f32) -> vec4<f32> {
  // Common initialization
  let r0 = length(rayOrigin);
  let theta0 = acos(clamp(rayOrigin.z / r0, -1.0, 1.0));
  let phi0 = atan2(rayOrigin.y, rayOrigin.x);
  
  // Star volumetric accumulation
  var accumulatedStarColor = vec3<f32>(0.0);
  var accumulatedStarOpacity = 0.0;
  
  let rs = M + sqrt(M * M - a * a);
  if (r0 < rs * 1.01) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  
  let t_closest = -dot(rayOrigin, rayDir);
  let closestPoint = rayOrigin + t_closest * rayDir;
  let impactParameter = length(closestPoint);
  if (impactParameter > diskRadius * 1.1) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  var state: GeodesicState;
  state.r = r0;
  state.theta = theta0;
  state.phi = phi0;
  
  let metric = computeMetric(r0, theta0, a, M);
  var sth0 = sin(theta0);
  if (sth0 < 1e-10) { sth0 = 1e-10; }
  
  let r_hat = normalize(rayOrigin);
  let theta_hat = vec3<f32>(-r_hat.z * r_hat.x, -r_hat.z * r_hat.y, r_hat.x * r_hat.x + r_hat.y * r_hat.y) / sqrt(r_hat.x * r_hat.x + r_hat.y * r_hat.y + 1e-10);
  let phi_hat = vec3<f32>(-r_hat.y, r_hat.x, 0.0) / sqrt(r_hat.x * r_hat.x + r_hat.y * r_hat.y + 1e-10);
  
  let rdot = dot(rayDir, r_hat);
  let thetadot = dot(rayDir, theta_hat);
  let phidot = dot(rayDir, phi_hat);
  
  let normalization = sqrt(metric.g_11 * rdot * rdot + metric.g_22 * thetadot * thetadot + metric.g_33 * phidot * phidot);
  state.ur = rdot / normalization * sqrt(metric.g_11);
  state.utheta = thetadot / normalization * sqrt(metric.g_22);
  state.uphi = phidot / normalization * sqrt(metric.g_33);
  if (abs(state.uphi) < 2e-3) {
    state.uphi = 2e-3 * sign(state.uphi);
  }
  
  // RK45 integration using shared function
  let atol = 1e-4;
  let rtol = 1e-4;
  let hmin = 1e-2;
  let hmax = 1.0;
  let maxSteps = 5000;
  
  var rk45_state: RK45State;
  rk45_state.h = 0.1;
  rk45_state.err_prev = 1.0;
  rk45_state.k1 = geodesicDerivatives(state, a, M);
  
  for (var step = 0; step < maxSteps; step++) {
    let prevZ = state.r * cos(state.theta);
    
    // Check if we're near the poles (theta close to 0 or pi)
    let poleThreshold = 0.15; // About 5.7 degrees from pole
    let nearPole = state.theta < poleThreshold || state.theta > (3.14159265359 - poleThreshold);
    
    var stepAccepted = false;
    if (nearPole) {
      // Use simple RK4 with larger step size near poles
      performRK4Step(&state, 0.01, a, M);
      stepAccepted = true;
    } else {
      stepAccepted = performRK45Step(&state, &rk45_state, a, M, atol, rtol, hmin, hmax);
    }
    
    if (stepAccepted) {
      // Step accepted - check termination conditions
      if (state.r < rs * 1.01 || state.r > maxDistance) {
        break;
      }
      
      // Sample star density
      let starSample = getStarDensity(state.r, state.theta, state.phi);
      if (starSample.density > 0.001) {
        let effectiveStepLength = rk45_state.h * length(vec3<f32>(rk45_state.k1.r, rk45_state.k1.theta * state.r, rk45_state.k1.phi * state.r * sin(state.theta)));
        let opticalDepth = starSample.density * effectiveStepLength * 0.5; // Use same multiplier as disk
        let transmission = exp(-opticalDepth);
        
        accumulatedStarOpacity += (1.0 - transmission) * (1.0 - accumulatedStarOpacity);
        accumulatedStarColor += starSample.emission * (1.0 - transmission) * (1.0 - accumulatedStarOpacity);
        
        // Early exit if star is opaque enough
        if (accumulatedStarOpacity > 0.95) {
          return vec4<f32>(accumulatedStarColor, 1.0);
        }
      }
      
      let currentZ = state.r * cos(state.theta);
      
      // Check for disk plane crossing
      if (prevZ * currentZ < 0.0) {
        let t = abs(prevZ) / (abs(prevZ) + abs(currentZ));
        let crossingR = mix(state.r - rk45_state.h * rk45_state.k1.r, state.r, t);
        let crossingTheta = mix(state.theta - rk45_state.h * rk45_state.k1.theta, state.theta, t);
        let crossingPhi = mix(state.phi - rk45_state.h * rk45_state.k1.phi, state.phi, t);
        let cylindricalRadius = crossingR * sin(crossingTheta);
        
        if (cylindricalRadius >= innerRadius && cylindricalRadius <= diskRadius) {
          let diskColor = renderThinDisk(cylindricalRadius, crossingPhi, innerRadius, diskRadius);
          // Composite star over disk if we have accumulated star color
          if (accumulatedStarOpacity > 0.01) {
            let finalColor = mix(diskColor.rgb, accumulatedStarColor, accumulatedStarOpacity);
            return vec4<f32>(finalColor, 1.0);
          }
          return diskColor;
        }
      }
    }
  }
  
  // Return accumulated star color if any, otherwise black
  if (accumulatedStarOpacity > 0.01) {
    return vec4<f32>(accumulatedStarColor, 1.0);
  }
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

fn traceGeodesicVolumetric(rayOrigin: vec3<f32>, rayDir: vec3<f32>, a: f32, M: f32, diskRadius: f32, innerRadius: f32, maxDistance: f32) -> vec4<f32> {
  // Common initialization (same as thin disk)
  let r0 = length(rayOrigin);
  let theta0 = acos(clamp(rayOrigin.z / r0, -1.0, 1.0));
  let phi0 = atan2(rayOrigin.y, rayOrigin.x);
    
  var accumulatedColor = vec3<f32>(0.0);
  var accumulatedOpacity = 0.0;
  
  let rs = M + sqrt(M * M - a * a);
  if (r0 < rs * 1.01) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
    
  let t_closest = -dot(rayOrigin, rayDir);
  let closestPoint = rayOrigin + t_closest * rayDir;
  let impactParameter = length(closestPoint);
  if (impactParameter > diskRadius * 1.5) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  var state: GeodesicState;
  state.r = r0;
  state.theta = theta0;
  state.phi = phi0;
    
  let metric = computeMetric(r0, theta0, a, M);
  var sth0 = sin(theta0);
  if (sth0 < 1e-10) { sth0 = 1e-10; }
    
  let r_hat = normalize(rayOrigin);
  let theta_hat = vec3<f32>(-r_hat.z * r_hat.x, -r_hat.z * r_hat.y, r_hat.x * r_hat.x + r_hat.y * r_hat.y) / sqrt(r_hat.x * r_hat.x + r_hat.y * r_hat.y + 1e-10);
  let phi_hat = vec3<f32>(-r_hat.y, r_hat.x, 0.0) / sqrt(r_hat.x * r_hat.x + r_hat.y * r_hat.y + 1e-10);
    
  let rdot = dot(rayDir, r_hat);
  let thetadot = dot(rayDir, theta_hat);
  let phidot = dot(rayDir, phi_hat);
    
  let normalization = sqrt(metric.g_11 * rdot * rdot + metric.g_22 * thetadot * thetadot + metric.g_33 * phidot * phidot);
  state.ur = rdot / normalization * sqrt(metric.g_11);
  state.utheta = thetadot / normalization * sqrt(metric.g_22);
  state.uphi = phidot / normalization * sqrt(metric.g_33);
  if (abs(state.uphi) < 2e-3) {
    state.uphi = 2e-3 * sign(state.uphi);
  }
  
  // RK45 integration using shared function
  let atol = 1e-4;
  let rtol = 1e-4;
  let hmin = 1e-2;
  let hmax = 0.2;
  let maxSteps = 2000;
  
  var rk45_state: RK45State;
  rk45_state.h = 0.1;
  rk45_state.err_prev = 1.0;
  rk45_state.k1 = geodesicDerivatives(state, a, M);
  
  // Volumetric sampling variables
  var inDiskRegion = false;
  var volumetricStepCounter = 0;
  let volumetricSampleRate = 1;
  
  for (var step = 0; step < maxSteps; step++) {
    // Check if we're near the poles (theta close to 0 or pi)
    let poleThreshold = 0.002; // About 5.7 degrees from pole
    let nearPole = state.theta < poleThreshold || state.theta > (3.14159265359 - poleThreshold);
    
    var stepAccepted = false;
    if (nearPole) {
    // if (false) {
      // Use simple RK4 with larger step size near poles
      performRK4Step(&state, 0.1, a, M);
      stepAccepted = true;
    } else {
      stepAccepted = performRK45Step(&state, &rk45_state, a, M, atol, rtol, hmin, hmax);
    }
    
    if (stepAccepted) {
      // Step accepted - check termination conditions
      if (state.r < rs * 1.01 || state.r > maxDistance) {
        break;
      }
      
      // Sample star density
      let starSample = getStarDensity(state.r, state.theta, state.phi);
      if (starSample.density > 0.001) {
        let effectiveStepLength = rk45_state.h * length(vec3<f32>(rk45_state.k1.r, rk45_state.k1.theta * state.r, rk45_state.k1.phi * state.r * sin(state.theta)));
        let opticalDepth = starSample.density * effectiveStepLength * 0.5; // Use same multiplier as disk
        let transmission = exp(-opticalDepth);
        
        accumulatedOpacity += (1.0 - transmission) * (1.0 - accumulatedOpacity);
        accumulatedColor += starSample.emission * (1.0 - transmission) * (1.0 - accumulatedOpacity);
        
        // Early exit if opaque enough
        if (accumulatedOpacity > 0.95) {
          return vec4<f32>(accumulatedColor, 1.0);
        }
      }
      
      // Volumetric sampling along the ray
      let currentZ = state.r * cos(state.theta);
      let cylindricalRadius = state.r * sin(state.theta);
      
      // Check if we're in the disk region
      let diskHeightMax = 1.0;
      inDiskRegion = cylindricalRadius >= innerRadius * 0.8 && 
                     cylindricalRadius <= diskRadius * 1.2 && 
                     abs(currentZ) < diskHeightMax;
      
      // Only sample volumetrics when in disk region and at sampling interval
      if (inDiskRegion) {
        volumetricStepCounter += 1;
        
        if (volumetricStepCounter % volumetricSampleRate == 0) {
          let sample = getDiskDensity(cylindricalRadius, currentZ, state.phi, innerRadius, diskRadius);
          
          if (sample.density > 0.001) {
            let effectiveStepLength = rk45_state.h * f32(volumetricSampleRate) * length(vec3<f32>(rk45_state.k1.r, rk45_state.k1.theta * state.r, rk45_state.k1.phi * state.r * sin(state.theta)));
            // let effectiveStepLength = rk45_state.h * f32(volumetricSampleRate);
            let opticalDepth = sample.density * effectiveStepLength * 0.5;
            let transmission = exp(-opticalDepth);

            accumulatedOpacity += (1.0 - transmission) * (1.0 - accumulatedOpacity);
            accumulatedColor += sample.emission * (1.0 - transmission) * (1.0 - accumulatedOpacity);

            if (accumulatedOpacity > 0.95) {
              break;
            }
          }
        }
      } else if (state.r > diskRadius * 1.5 && state.ur > 0.0) {
        break; // Ray is moving away from disk
      }
    }
  }
    
  return vec4<f32>(accumulatedColor, 1.0);
}


@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  // Block-based rendering - compute only every nth pixel, use for entire block
  let blockSize = 1; // Increased for better performance
  let pixelX = i32(fragCoord.x);
  let pixelY = i32(fragCoord.y);
  let blockX = (pixelX / blockSize) * blockSize;
  let blockY = (pixelY / blockSize) * blockSize;
    
  // Use the center of the block for computation
  let centerX = f32(blockX + blockSize / 2) + 0.2;
  let centerY = f32(blockY + blockSize / 2);
    
  let blockScreenPos = vec2<f32>(
      (centerX / uniforms.screenWidth) * 2.0 - 1.0,
      (centerY / uniforms.screenHeight) * 2.0 - 1.0
                                 );
    
  // Use block center for ray computation
  let aspectRatio = uniforms.screenWidth / uniforms.screenHeight;
  let ndc = vec2<f32>(blockScreenPos.x * aspectRatio, blockScreenPos.y);
    
  let fov = 0.785398;
  let rayDirLocal = normalize(vec3<f32>(ndc.x * tan(fov/2.0), ndc.y * tan(fov/2.0), -1.0));
    
  let viewMatrixInv = transpose(uniforms.viewMatrix);
  let rayDir = normalize((viewMatrixInv * vec4<f32>(rayDirLocal, 0.0)).xyz);
    
  // Choose rendering mode based on toggle
  if (uniforms.volumetricMode > 0.5) {
    // Volumetric rendering
    let volumetricResult = traceGeodesicVolumetric(uniforms.cameraPos, rayDir, uniforms.blackHoleSpin, uniforms.blackHoleMass, uniforms.diskRadius, uniforms.innerRadius, uniforms.observerDistance * 1.3);
    return volumetricResult;
  } else {
    // Thin disk rendering
    let thinDiskResult = traceGeodesicThinDisk(uniforms.cameraPos, rayDir, uniforms.blackHoleSpin, uniforms.blackHoleMass, uniforms.diskRadius, uniforms.innerRadius, uniforms.observerDistance * 1.3);
    return thinDiskResult;
  }
}
