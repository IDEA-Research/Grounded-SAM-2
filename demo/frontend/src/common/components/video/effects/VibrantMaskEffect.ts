/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import BaseGLEffect from '@/common/components/video/effects/BaseGLEffect';
import {
  EffectFrameContext,
  EffectInit,
} from '@/common/components/video/effects/Effect';
import vertexShaderSource from '@/common/components/video/effects/shaders/DefaultVert.vert?raw';
import fragmentShaderSource from '@/common/components/video/effects/shaders/VibrantMask.frag?raw';
import {Tracklet} from '@/common/tracker/Tracker';
import {
  generateLUTDATA,
  load3DLUT,
  preAllocateTextures,
} from '@/common/utils/ShaderUtils';
import {RLEObject, decode} from '@/jscocotools/mask';
import invariant from 'invariant';
import {CanvasForm} from 'pts';

export default class VibrantMaskEffect extends BaseGLEffect {
  private lutSize: number = 4;

  private _numMasks: number = 0;
  private _numMasksUniformLocation: WebGLUniformLocation | null = null;
  private _currentFrameLocation: WebGLUniformLocation | null = null;
  private _lutTextures: WebGLTexture[] = [];
  private _maskTextures: WebGLTexture[] = [];

  // Must be 1, main background texture takes 0.
  private _extraTextureUnit: number = 1;

  // Must from start 2, main texture takes 0 and 1.
  private _masksTextureUnitStart: number = 2;

  constructor() {
    super(3);
    this.vertexShaderSource = vertexShaderSource;
    this.fragmentShaderSource = fragmentShaderSource;
  }

  protected setupUniforms(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    init: EffectInit,
  ): void {
    super.setupUniforms(gl, program, init);

    gl.uniform1i(
      gl.getUniformLocation(program, 'uColorGradeLUT'),
      this._extraTextureUnit,
    );

    this._numMasksUniformLocation = gl.getUniformLocation(program, 'uNumMasks');
    gl.uniform1i(this._numMasksUniformLocation, this._numMasks);

    this._currentFrameLocation = gl.getUniformLocation(
      program,
      'uCurrentFrame',
    );
    gl.uniform1f(this._currentFrameLocation, 0);

    // We know the max number of textures, pre-allocate 3.
    this._maskTextures = preAllocateTextures(gl, 3);

    this._lutTextures = []; // clear any previous pool of textures

    for (let i = 0; i < this.numVariants; i++) {
      const _lutData = generateLUTDATA(this.lutSize);
      const _extraTexture = load3DLUT(gl, this.lutSize, _lutData);
      this._lutTextures.push(_extraTexture as WebGLTexture);
    }
  }

  apply(form: CanvasForm, context: EffectFrameContext, _tracklets: Tracklet[]) {
    const gl = this._gl;
    const program = this._program;

    if (!program) {
      return;
    }
    invariant(gl !== null, 'WebGL2 context is required');

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // dynamic uniforms per frame
    gl.uniform1f(this._currentFrameLocation, context.frameIndex);
    gl.uniform1i(this._numMasksUniformLocation, context.masks.length);

    // Bind the LUT texture to texture unit 1
    const lutTexture = this._lutTextures[this.variant];
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, lutTexture);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._frameTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      context.width,
      context.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      context.frame,
    );

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    // Create and bind 2D textures for each mask
    context.masks.forEach((mask, index) => {
      const decodedMask = decode([mask.bitmap as RLEObject]);
      const maskData = decodedMask.data as Uint8Array;
      gl.activeTexture(gl.TEXTURE0 + index + this._masksTextureUnitStart);
      gl.bindTexture(gl.TEXTURE_2D, this._maskTextures[index]);

      // dynamic uniforms per mask
      gl.uniform1i(
        gl.getUniformLocation(program, `uMaskTexture${index}`),
        this._masksTextureUnitStart + index,
      );

      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.LUMINANCE,
        context.height,
        context.width,
        0,
        gl.LUMINANCE,
        gl.UNSIGNED_BYTE,
        maskData,
      );
    });

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Unbind textures
    gl.bindTexture(gl.TEXTURE_2D, null);
    context.masks.forEach((_, index) => {
      gl.activeTexture(gl.TEXTURE0 + index + this._masksTextureUnitStart);
      gl.bindTexture(gl.TEXTURE_2D, null);
    });

    const ctx = form.ctx;
    invariant(this._canvas !== null, 'canvas is required');
    ctx.drawImage(this._canvas, 0, 0);
  }

  async cleanup(): Promise<void> {
    super.cleanup();

    if (this._gl != null) {
      // Delete mask textures to prevent memory leaks
      this._maskTextures.forEach(texture => {
        if (texture != null && this._gl != null) {
          this._gl.deleteTexture(texture);
        }
      });
      this._maskTextures = [];
    }
  }
}
