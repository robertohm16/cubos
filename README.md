# cubos<script id="vertexShader_buffer" type="x-shader/x-vertex">attribute vec4 a_position;  
  uniform mat4 u_modelViewMatrix;
  uniform mat4 u_projectionMatrix;
  
  void main() {
    gl_Position = a_position;
  }
</script>
<script id="fragmentShader_under" type="x-shader/x-vertex">
  #extension GL_EXT_shader_texture_lod : enable
  precision highp float;
  
  uniform vec2 u_resolution;
  uniform float u_time;
  uniform vec2 u_mouse;
  uniform sampler2D u_noise;
  
  uniform vec2 u_cam;
  
  uniform samplerCube u_environment;
  
  /* Raymarching constants */
  /* --------------------- */
  const float MAX_TRACE_DISTANCE = 8.;             // max trace distance
  const float INTERSECTION_PRECISION = 0.001;       // precision of the intersection
  const int NUM_OF_TRACE_STEPS = 100;               // max number of trace steps
  const float STEP_MULTIPLIER = .8;                 // the step mutliplier - ie, how much further to progress on each step
  
  /* Structures */
  /* ---------- */
  struct Camera {
    vec3 ro;
    vec3 rd;
    vec3 forward;
    vec3 right;
    vec3 up;
    float FOV;
  };
  struct Surface {
    float len;
    vec3 position;
    vec3 colour;
    float id;
    float steps;
    float AO;
  };
  struct Model {
    float dist;
    vec3 colour;
    float id;
  };
  
  vec2 getScreenSpace() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.y, u_resolution.x);
    
    return uv;
  }
  
  float easeInOutCubic(float d) {
    if (d < 0.5) return 4. * pow(d, 3.);
    return 1. - pow(-2. * d + 2., 3.) * .5;
  }
  float easeInOutExpo(float k) {
    if(k == 0.) return 0.;
    if(k == 1.) return 1.;
    k *= 2.;
    if(k < 1.) return .5 * pow(1024., k - 1.);
    return .5 * ( -pow(2., -10. * (k - 1.)) + 2. );
  }
  
  /*--------------------------------
  /  Modelling
  / -------------------------------- */
  float smin(float a, float b, float k) {
      float res = exp(-k*a) + exp(-k*b);
      return -log(res)/k;
  }

  mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                  oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                  oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                  0.0,                                0.0,                                0.0,                                1.0);
}
  
  float udBox( vec3 p, vec3 b ) {
    return length(max(abs(p)-b,0.0));
  }
  
  float t = 0.;
  Model model(vec3 p) {
    float d = length(p) - .2;
    // vec3 colour = vec3(1,0,0);
    
    d = 100.;
    vec3 colour2 = vec3(0);
    vec3 colour1 = vec3(0);
    vec3 colour = vec3(1);
    
    vec3 pos1, pos2 = vec3(0,0,0);
    vec3 endpos1 = vec3(-.25, 0, 0);
    vec3 endpos2 = vec3(.25, 0, 0);
    
    pos1 = mix(vec3(0), endpos1, t);
    pos2 = mix(vec3(0), endpos2, t);
    
    vec3 _p = p;
    
    for(float i = 0.; i < 3.; i++) {
      vec3 c;
      vec3 pos = vec3(0);
      if(i == 1.) {
        pos = pos1;
        c = colour2;
      }
      if(i == 2.) {
        pos = pos2;
        c = colour1;
      }
      p = _p + pos;
      float t5 = u_time / 5. * (i + 1.);
      p = (rotationMatrix(vec3(cos(t5), sin(t5), .5), u_time / 3. ) * vec4(p, 1.)).xyz;
      if(d == 100.) {
        d = udBox(p, vec3(.1));
      } else {
        float d1 = udBox(p, vec3(.1));
        float ddiff = (d1 - d)/d1;
        d = smin(d, d1, mix(90., 20., t));
        colour = mix(c, colour, smoothstep(0., 1., ddiff * .7)); // This mixes the colours together based on the gradient produced by the smin
      }
    }
    
    return Model(d, colour, 1.);
  }
  Model map( vec3 p ){
    return model(p);
  }
  
  Surface calcIntersection( in Camera cam ){
    float h =  INTERSECTION_PRECISION*2.0;
    float rayDepth = 0.0;
    float hitDepth = -1.0;
    float id = -1.;
    float steps = 0.;
    float ao = 0.;
    vec3 position;
    vec3 colour;

    for( int i=0; i< NUM_OF_TRACE_STEPS ; i++ ) {
      if( abs(h) < INTERSECTION_PRECISION || rayDepth > MAX_TRACE_DISTANCE ) break;
      position = cam.ro+cam.rd*rayDepth;
      Model m = map( position );
      h = m.dist;
      rayDepth += h * STEP_MULTIPLIER;
      id = m.id;
      steps += 1.;
      ao += max(h, 0.);
      colour = m.colour;
    }

    if( rayDepth < MAX_TRACE_DISTANCE ) hitDepth = rayDepth;
    if( rayDepth >= MAX_TRACE_DISTANCE ) id = -1.0;

    return Surface( hitDepth, position, colour, id, steps, ao );
  }
  Camera getCamera(in vec2 uv, in vec3 pos, in vec3 target) {
    vec3 forward = normalize(target - pos);
    vec3 right = normalize(vec3(forward.z, 0., -forward.x));
    vec3 up = normalize(cross(forward, right));
    
    float FOV = .6;
    
    return Camera(
      pos,
      normalize(forward + FOV * uv.x * right + FOV * uv.y * up),
      forward,
      right,
      up,
      FOV
    );
  }
  
  
  float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax ) {
    float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ ) {
      float h = map( ro + rd*t ).dist;
      res = min( res, 8.0*h/t );
      t += clamp( h, 0.02, 0.10 );
      if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
  }
  float calcAO( in vec3 pos, in vec3 nor ) {
    float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
      float hr = 0.01 + 0.12*float(i)/4.0;
      vec3 aopos =  nor * hr + pos;
      float dd = map( aopos ).dist;
      occ += -(dd-hr)*sca;
      sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
  }
  // This is here to provide a fallback for devices that don't support GL_EXT_shader_texture_lod
  vec4 cubeTexture(samplerCube map, vec3 uv, float lod) {
    #ifdef GL_EXT_shader_texture_lod
    return textureCubeLodEXT(map, uv, lod);
    #endif
    return textureCube(map, uv);
  }
  vec3 shade(Surface surface, vec3 nor, vec3 ref, Camera cam) {
    
    vec3 col = surface.colour;
    vec3 pos = surface.position;
    
    vec3 I = normalize(pos - cam.ro);
    vec3 R = reflect(I, nor);
    vec3 reflection = cubeTexture(u_environment, R, 5.).rgb;
    // reflection *= 0.;
    // col = reflection;
    // lighitng        
    float occ = 1./surface.AO;
    vec3  lig = normalize( vec3(-0.6, 0.7, -0.) );
    float amb = clamp( 0.5+0.5*nor.y, 0.0, 1.0 );
    float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
    float bac = clamp( dot( nor, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
    // float dom = smoothstep( -0.1, 0.1, ref.y );
    float fre = pow( clamp(1.0+dot(nor,cam.rd),0.0,1.0), 2.0 );
    float spe = pow(clamp( dot( ref, lig ), 0.0, 1.0 ),4.0);

    // dif *= softshadow( pos, lig, 0.02, 2.5 );
    //dom *= softshadow( pos, ref, 0.02, 2.5 );

    vec3 lin = vec3(0.0);
    lin += 1.20*dif*vec3(.95,0.80,0.60);
    lin += 1.20*spe*vec3(1.00,0.85,0.55)*dif;
    lin += 0.80*amb*vec3(0.50,0.70,.80)*occ;
    //lin += 0.30*dom*vec3(0.50,0.70,1.00)*occ;
    lin += 0.30*bac*vec3(0.25,0.25,0.25)*occ;
    lin += 0.20*fre*vec3(1.00,1.00,1.00)*occ;
    col = col*lin;
    
    col += reflection * .5;

    return col;
  }
  
  // Calculates the normal by taking a very small distance,
  // remapping the function, and getting normal for that
  vec3 calcNormal( in vec3 pos ){
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
      map(pos+eps.xyy).dist - map(pos-eps.xyy).dist,
      map(pos+eps.yxy).dist - map(pos-eps.yxy).dist,
      map(pos+eps.yyx).dist - map(pos-eps.yyx).dist );
    return normalize(nor);
  }
  
  vec3 render(Surface surface, Camera cam, vec2 uv) {
    vec3 colour = vec3(.4,.4,.45);
    vec3 colourB = vec3(.1, .1, .1);
    
    colour = mix(colourB, colour, smoothstep(1., 0., (length(uv) - surface.steps/100.) * (1.+smoothstep(-1., -1.5, -abs(cam.ro.z))*.5)));
    
    // colour = mix(vec3(.5,0,.3), vec3(0,.5,0), smoothstep(-1., -1.5, cam.ro.z))+surface.steps/100.;
    
    // colour -= texture2D(u_noise, uv).rgb;
    
    if (surface.id == 1.){
      vec3 surfaceNormal = calcNormal( surface.position );
      vec3 ref = reflect(cam.rd, surfaceNormal);
      colour = surfaceNormal;
      colour = shade(surface, surfaceNormal, ref, cam);
    }

    return colour;
  }
  
  void main() {
    vec2 uv = getScreenSpace();
    
    t = easeInOutExpo( smoothstep(0., 1., sin(u_time * .25) * 2.) );
    
    float camd = mix(.8, 1.5, t);
    
    float c = cos(u_cam.x);
    float s = sin(u_cam.x);
    mat3 xrot = mat3(
      c, 0, s,
      0, 1, 0,
      -s, 0, c
    );
    // c = cos(u_cam.y);
    // s = sin(u_cam.y);
    // mat3 yrot = mat3(
    //   1, 0, 0,
    //   0, c, -s,
    //   0, s, c
    // );
    // xrot *= yrot;
    
    vec3 campos = vec3( 0, 0, camd ) * xrot;
    // campos += vec3( 0, cos(u_cam.y)*camd, sin(u_cam.y)*camd );
    
    // Camera cam = getCamera(uv, mix(vec3(0,0,-.8), vec3(0,0,-1.5), t), vec3(0));
    // Camera cam = getCamera(uv, vec3(0.,0.,1.25), vec3(0, 0, 0));
    Camera cam = getCamera(uv, campos, vec3(0));
    
    Surface surface = calcIntersection(cam);
    
    gl_FragColor = vec4(render(surface, cam, uv), 1.);
    // gl_FragColor = vec4(vec3(surface.steps/100.), 1.);
  }
</script>
