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

fn computeMetric(r: f32, th: f32, a: f32, M: f32) -> BoyerLindquistMetric {
  var metric: BoyerLindquistMetric;
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

fn traceGeodesic(rayOrigin: vec3<f32>, rayDir: vec3<f32>, a: f32, M: f32, diskRadius: f32, innerRadius: f32, maxDistance: f32) -> DiskHit {
  let r0 = length(rayOrigin);
  let theta0 = acos(clamp(rayOrigin.z / r0, -1.0, 1.0));
  let phi0 = atan2(rayOrigin.y, rayOrigin.x);
    
  var result: DiskHit;
  result.hit = false;
  result.r = 0.0;
  result.phi = 0.0;
  
  let rs = M + sqrt(M * M - a * a); // Event horizon
  if (r0 < rs * 1.01) {
    return result;
  }
    
  // Calculate impact parameter
  let t_closest = -dot(rayOrigin, rayDir);
  let closestPoint = rayOrigin + t_closest * rayDir;
  let impactParameter = length(closestPoint);

  // Check if the ray is too far from the disk or too close to the black hole
  if (impactParameter > diskRadius * 1.1) {
    return result; // Too far from the disk
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
  if (abs(state.uphi) < 2e-3) {
    state.uphi = 2e-3 * sign(state.uphi); // Prevent zero momentum in phi direction
  }
    
  // RK45 Dormand-Prince integration with adaptive stepping
  var h = 0.2; // Initial step size
  let atol = 1e-4;
  let rtol = 1e-4;
  let hmin = 1e-2;
  let hmax = 1.0;
  let maxSteps = 5000;
    
  // Precomputed RK45 Dormand-Prince coefficients (stored once)
  const c2 = 0.2;
  const c3 = 0.3;
  const c4 = 0.8;
  const c5 = 0.888888888889;
    
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
            
      // Early termination conditions for performance
      if (state.r < rs * 1.01) {
        return result; // Hit event horizon
      }
            
      if (state.r > maxDistance) {
        return result; // Escaped to infinity
      }
            
      // Early exit if ray is moving away from disk plane and far from it
      let currentZ = state.r * cos(state.theta);
      if (state.ur > 0 && state.r > diskRadius * 1.2) {
        return result; // Moving away from the black hole, won't hit
      }

      // Check for disk crossing
      if (prevZ * currentZ < 0.0) {
        // Ray crosses the disk plane - interpolate to find exact crossing point
        let t = abs(prevZ) / (abs(prevZ) + abs(currentZ));
        
        // Interpolate position at crossing
        let crossingR = mix(state.r - h * k1.r, state.r, t);
        let crossingTheta = mix(state.theta - h * k1.theta, state.theta, t);
        let crossingPhi = mix(state.phi - h * k1.phi, state.phi, t);
        
        // Convert to cylindrical radius for disk check
        let cylindricalRadius = crossingR * sin(crossingTheta);
        
        // Check if the crossing point is within the disk bounds
        if (cylindricalRadius >= innerRadius && cylindricalRadius <= diskRadius) {
          result.hit = true;
          result.r = cylindricalRadius;
          result.phi = crossingPhi;
          return result;
        }
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
    
  return result;
}


@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  // Block-based rendering - compute only every nth pixel, use for entire block
  let blockSize = 1; // Reduced from 16 for better quality
  let pixelX = i32(fragCoord.x);
  let pixelY = i32(fragCoord.y);
  let blockX = (pixelX / blockSize) * blockSize;
  let blockY = (pixelY / blockSize) * blockSize;
    
  // Use the center of the block for computation
  let centerX = f32(blockX + blockSize / 2);
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
    
  // General relativistic ray tracing for entire screen
  let diskHit = traceGeodesic(uniforms.cameraPos, rayDir, uniforms.blackHoleSpin, uniforms.blackHoleMass, uniforms.diskRadius, uniforms.innerRadius, uniforms.observerDistance * 1.3);
  
  if (diskHit.hit) {
    // Create texture coordinates from hit position
    // Scale phi to [0, 1] range for periodic wrapping
    let phiNormalized = (diskHit.phi + 3.14159265359) / 6.28318530718;
    let texCoord = vec2<f32>(diskHit.r * 0.2, phiNormalized);
    
    // Multiple periods for different frequency components
    let period1 = vec2<f32>(20.0, 1.0);  // Medium radial period
    let period2 = vec2<f32>(8.0, 1.0);   // Smaller radial period
    let period3 = vec2<f32>(3.0, 1.0);   // Fine radial details
    
    // Generate multiple octaves of periodic noise with different radial scales
    let noise1 = fractalPeriodicNoise(texCoord * 2.0, period1, 4);
    let noise2 = fractalPeriodicNoise(texCoord * 5.0, period2, 3);
    let noise3 = fractalPeriodicNoise(texCoord * 15.0, period3, 2);
    let noise4 = periodicNoise2D(texCoord * 40.0, vec2<f32>(1.5, 1.0));
    
    // Create spiral patterns by mixing radial and angular components
    let spiralCoord = vec2<f32>(diskHit.r * 0.15, phiNormalized * 3.0 + diskHit.r * 0.1);
    let spiralNoise = fractalPeriodicNoise(spiralCoord, vec2<f32>(10.0, 1.0), 3);
    
    // Radial turbulence with fractal characteristics
    let radialTurbulence = fractalNoise(vec2<f32>(diskHit.r * 0.3, phiNormalized * 10.0), 4);
    
    // Combine noises for complex, fractal-like texture
    let turbulence = noise1 * 0.3 + noise2 * 0.25 + noise3 * 0.2 + noise4 * 0.15 + spiralNoise * 0.05 + radialTurbulence * 0.05;
    
    // Create color variations - hot accretion disk
    let heat = 0.7 + turbulence * 0.3;
    let baseColor = vec3<f32>(heat * 1.2, heat * 0.2, heat * 0.1);
    
    // Add bright hot spots with more complex patterns
    let hotSpots1 = max(0.0, noise1 - 0.4) * 1.5;
    let hotSpots2 = max(0.0, noise2 - 0.5) * 1.0;
    let hotSpots3 = max(0.0, spiralNoise - 0.3) * 0.8;
    let totalHotSpots = hotSpots1 + hotSpots2 + hotSpots3;
    
    let color = baseColor + vec3<f32>(totalHotSpots, totalHotSpots * 0.7, totalHotSpots * 0.2);
    
    // Radial falloff for more realistic appearance
    let radialFactor = 1.0 - pow((diskHit.r - uniforms.innerRadius) / (uniforms.diskRadius - uniforms.innerRadius), 2.0);
    let finalColor = color * (0.5 + radialFactor * 0.5);
    
    return vec4<f32>(finalColor, 1.0);
  } else {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // Black background
  }
}
