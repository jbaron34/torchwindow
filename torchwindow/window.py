import ctypes
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=UserWarning)
    import sdl2

from sdl2 import video
from OpenGL import GL as gl
from cuda import cudart as cu

from .shaders import create_shader_program
from .exceptions import SDLException, CudaException, OpenGLException

import logging

logger = logging.getLogger(__name__)


class Window:
    def __init__(self, width: int = 800, height: int = 600, name: str = "torchwindow"):
        self.name = name
        self.width = width
        self.height = height

        self.cuda_is_setup = False
        self.running = True

        self.setup()

    def setup_sdl(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO):
            raise SDLException(sdl2.SDL_GetError())

        self.sdl_window = sdl2.SDL_CreateWindow(
            self.name.encode(),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            self.width,
            self.height,
            sdl2.SDL_WINDOW_OPENGL,
        )
        if not self.sdl_window:
            raise SDLException(sdl2.SDL_GetError())

        # Force OpenGL 3.3 'core' context.
        # Must set *before* creating GL context!
        video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MINOR_VERSION, 3)
        video.SDL_GL_SetAttribute(
            video.SDL_GL_CONTEXT_PROFILE_MASK, video.SDL_GL_CONTEXT_PROFILE_CORE
        )
        self.gl_context = sdl2.SDL_GL_CreateContext(self.sdl_window)

    def setup_opengl(self):
        self.shader_program = create_shader_program()
        self.vao = gl.glGenVertexArrays(1)

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            self.width,
            self.height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def setup_cuda(self):
        if self.cuda_is_setup:
            return

        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise SDLException(sdl2.SDL_GetError())

        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise OpenGLException(
                "OpenGL context may be running on integrated graphics"
            )

        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to register opengl texture")

        self.cuda_is_setup = True

    def render(self):
        gl.glUseProgram(self.shader_program)
        try:
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
        sdl2.SDL_GL_SwapWindow(self.sdl_window)

    def step(self):
        event = sdl2.SDL_Event()
        if self.running:
            while sdl2.SDL_PollEvent(ctypes.byref(event)):
                if (
                    event.type == sdl2.SDL_WINDOWEVENT
                    and event.window.event == sdl2.SDL_WINDOWEVENT_CLOSE
                ):
                    self.running = False
            self.render()

    def setup(self):
        self.setup_sdl()
        self.setup_opengl()
        self.setup_cuda()

    def draw(self, tensor):
        if not self.running:
            return
        if not self.cuda_is_setup:
            self.setup_cuda()
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to get mapped array")
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            tensor.data_ptr(),
            4 * 4 * self.width,
            4 * 4 * self.width,
            self.height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to copy from tensor to texture")

        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to unmap graphics resource")
        self.step()

    def close(self):
        self.running = False
        sdl2.SDL_GL_DeleteContext(self.gl_context)
        sdl2.SDL_DestroyWindow(self.sdl_window)
        sdl2.SDL_Quit()
