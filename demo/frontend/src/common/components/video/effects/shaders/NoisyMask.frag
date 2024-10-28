#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

precision mediump float;

in vec2 vTexCoord;

uniform float uCurrentFrame;
uniform int uNumMasks;
uniform sampler2D uMaskTexture0;
uniform sampler2D uMaskTexture1;
uniform sampler2D uMaskTexture2;

out vec4 fragColor;

vec3 startColor = vec3(0.0f, 0.67f, 1.0f);
vec3 endColor = vec3(0.05f, 0.06f, 0.05f);

float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898f, 78.233f))) *
    43758.5453123f);
}

void main() {
  vec4 finalColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float totalMaskValue = 0.0f;

  if(uNumMasks > 0) {
    float maskValue0 = texture(uMaskTexture0, vec2(vTexCoord.y, vTexCoord.x)).r;
    totalMaskValue += maskValue0;
  }
  if(uNumMasks > 1) {
    float maskValue1 = texture(uMaskTexture1, vec2(vTexCoord.y, vTexCoord.x)).r;
    totalMaskValue += maskValue1;
  }
  if(uNumMasks > 2) {
    float maskValue2 = texture(uMaskTexture2, vec2(vTexCoord.y, vTexCoord.x)).r;
    totalMaskValue += maskValue2;
  }

  // Dynamic color alteration using sin(time)
  float time = uCurrentFrame * 0.1f;
  vec3 dynamicColor = mix(startColor, endColor, sin(time));
  vec3 colorVariation = mix(vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f), vTexCoord.y);

  // apply randomness to the final color
  float rnd = random(vTexCoord.xy);
 
  if(totalMaskValue > 0.0f) {
    finalColor = vec4(mix(dynamicColor, colorVariation, rnd), 1.0f);
  } else {
    finalColor.a = 0.0f;
  }
  fragColor = finalColor;
}