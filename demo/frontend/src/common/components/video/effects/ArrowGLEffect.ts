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
import fragmentShaderSource from '@/common/components/video/effects/shaders/Arrow.frag?raw';
import vertexShaderSource from '@/common/components/video/effects/shaders/DefaultVert.vert?raw';
import {Tracklet} from '@/common/tracker/Tracker';
import {normalizeBounds} from '@/common/utils/ShaderUtils';
import {RLEObject, decode} from '@/jscocotools/mask';
import invariant from 'invariant';
import {CanvasForm} from 'pts';

export default class ArrowGLEffect extends BaseGLEffect {
  private _numMasks: number = 0;
  private _numMasksUniformLocation: WebGLUniformLocation | null = null;

  // Must from start 1, main texture takes.
  private _masksTextureUnitStart: number = 1;

  constructor() {
    super(4);
    this.vertexShaderSource = vertexShaderSource;
    this.fragmentShaderSource = fragmentShaderSource;
  }

  protected setupUniforms(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    init: EffectInit,
  ): void {
    super.setupUniforms(gl, program, init);

    this._numMasksUniformLocation = gl.getUniformLocation(program, 'uNumMasks');
    gl.uniform1i(this._numMasksUniformLocation, this._numMasks);
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
    const styleIndex = Math.floor(this.variant / 2) % 2;
    gl.uniform1i(this._numMasksUniformLocation, context.masks.length);
    gl.uniform1f(
      gl.getUniformLocation(program, 'uCurrentFrame'),
      context.frameIndex,
    );
    gl.uniform1i(
      gl.getUniformLocation(program, 'uLineColor'),
      this.variant % 2 === 0 ? 0 : 1,
    );
    gl.uniform1i(
      gl.getUniformLocation(program, 'uArrow'),
      styleIndex === 0 ? 1 : 0,
    );

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
      const maskTexture = gl.createTexture();
      const decodedMask = decode([mask.bitmap as RLEObject]);
      const maskData = decodedMask.data as Uint8Array;
      gl.activeTexture(gl.TEXTURE0 + index + this._masksTextureUnitStart);
      gl.bindTexture(gl.TEXTURE_2D, maskTexture);

      const boundaries = normalizeBounds(
        mask.bounds[0],
        mask.bounds[1],
        context.width,
        context.height,
      );

      gl.uniform1i(
        gl.getUniformLocation(program, `uMaskTexture${index}`),
        index + this._masksTextureUnitStart,
      );
      gl.uniform4fv(gl.getUniformLocation(program, `bbox${index}`), boundaries);

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
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    });

    const ctx = form.ctx;
    invariant(this._canvas !== null, 'canvas is required');
    ctx.drawImage(this._canvas, 0, 0);
  }
}
