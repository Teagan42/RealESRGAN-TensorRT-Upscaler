import os
import cv2
import torch
import numpy as np
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("realesrgan")
format_logger = logging.getLogger("format_validate")

class RealESRGANPipeline:
    def __init__(self, model_path: str, scale_factor: int = 4, device: str = "cuda"):
        """
        Initialize the RealESRGAN pipeline.
        
        Args:
            model_path: Path to the TensorRT model
            scale_factor: 2 for 2X model, 4 for 4X model
            device: Device to run inference on (cuda or cpu)
        """
        self.model_path = Path(model_path)
        self.scale_factor = scale_factor
        self.device = device
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Load TensorRT model
        try:
            from torch2trt import TRTModule
            self.model_trt = TRTModule()
            self.model_trt.load_state_dict(torch.load(str(self.model_path)))
            logger.info(f"Loaded TensorRT model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load TensorRT model: {e}")
            raise
            
    def preprocess_image(self, img_path: Union[str, Path]) -> torch.Tensor:
        """Preprocess an image for inference"""
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load and preprocess the input image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image from {img_path}")
        
        return self._preprocess_array(img)
        
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a video frame for inference"""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
            
        return self._preprocess_array(frame)
        
    def _preprocess_array(self, img_array: np.ndarray) -> torch.Tensor:
        """Common preprocessing for both images and video frames"""
        # Convert BGR to RGB and normalize
        img = img_array[..., ::-1].transpose(2, 0, 1) / 255.0
        x = torch.from_numpy(img).cuda().half().unsqueeze(0)
        
        if self.scale_factor == 2:
            # Special preprocessing for 2X model
            b, c, h, w = x.size()
            h = h // 2
            w = w // 2
            x = x.view(b, c, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, 12, h, w)
            
        return x

    def run_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference with the TensorRT model"""
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                y_trt = self.model_trt(x)
                torch.cuda.synchronize()
            return y_trt
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def postprocess(self, y_trt: torch.Tensor) -> np.ndarray:
        """Convert the output tensor to an image"""
        y_trt_np = y_trt.squeeze(0).cpu().detach().numpy()
        y_trt_np = y_trt_np.transpose(1, 2, 0)
        y_trt_np = (y_trt_np * 255).clip(0, 255).astype(np.uint8)
        
        # Convert back to BGR for OpenCV
        return y_trt_np[..., ::-1]
        
    def process_image(self, img_path: Union[str, Path]) -> np.ndarray:
        """Process a single image from start to finish"""
        x = self.preprocess_image(img_path)
        y = self.run_inference(x)
        return self.postprocess(y)
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame from start to finish"""
        x = self.preprocess_frame(frame)
        y = self.run_inference(x)
        return self.postprocess(y)
    
    def process_video(self, input_path: Union[str, Path], output_path: Union[str, Path], 
                     max_frames: Optional[int] = None, ensure_compatibility: bool = True) -> None:
        """
        Process a complete video file
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
            max_frames: Maximum number of frames to process (None for all)
            ensure_compatibility: Whether to ensure cloud compatibility
        """
        input_path, output_path = Path(input_path), Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Video has invalid metadata: {input_path}")

        frame_limit = max_frames if max_frames is not None else float("inf")
        silent_path = output_path.with_name(output_path.stem + ".video-only.mp4")
        mux_path = output_path.with_name(output_path.stem + ".muxed" + output_path.suffix)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width * self.scale_factor}x{height * self.scale_factor}",
            "-r", str(fps), "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(silent_path),
        ]
        try:
            ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            cap.release()
            raise
        count = 0
        try:
            while count < frame_limit:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.process_frame(frame)
                if result.shape[:2] != (height * self.scale_factor, width * self.scale_factor):
                    raise ValueError(
                        f"Model output is {result.shape[1]}x{result.shape[0]}, expected "
                        f"{width * self.scale_factor}x{height * self.scale_factor}"
                    )
                assert ffmpeg.stdin is not None
                ffmpeg.stdin.write(result.tobytes())
                count += 1
                if count % 100 == 0:
                    logger.info("Processed %d frames", count)
        except BaseException:
            if ffmpeg.stdin:
                ffmpeg.stdin.close()
            ffmpeg.kill()
            ffmpeg.wait()
            silent_path.unlink(missing_ok=True)
            raise
        finally:
            cap.release()
        assert ffmpeg.stdin is not None
        ffmpeg.stdin.close()
        error = ffmpeg.stderr.read() if ffmpeg.stderr else b""
        if ffmpeg.wait() != 0:
            silent_path.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg video encoding failed: {error.decode(errors='replace')}")
        if count == 0:
            silent_path.unlink(missing_ok=True)
            raise ValueError(f"No frames could be decoded from {input_path}")

        # Re-encode audio to AAC for broad compatibility; optional mapping allows
        # silent source videos to work. Limit both streams to the frames written.
        mux_command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(silent_path), "-i", str(input_path),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
            "-t", f"{count / fps:.9f}", "-movflags", "+faststart", str(mux_path),
        ]
        try:
            result = subprocess.run(mux_command, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(f"FFmpeg audio mux failed: {result.stderr.strip()}")
            os.replace(mux_path, output_path)
        finally:
            silent_path.unlink(missing_ok=True)
            mux_path.unlink(missing_ok=True)
        logger.info("Processed %d frames; output with source audio saved to %s", count, output_path)

    @staticmethod
    def extract_frames_from_video(video_path: Union[str, Path], 
                                 max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float, int, int]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (None for all)
            
        Returns:
            Tuple containing:
            - List of frames as numpy arrays
            - FPS of the video
            - Width of the video
            - Height of the video
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is not None:
            frames_to_extract = min(max_frames, total_frames)
            logger.info(f"Extracting {frames_to_extract} frames out of {total_frames}")
        else:
            frames_to_extract = total_frames
            logger.info(f"Extracting all {total_frames} frames")

        frames_list = []
        frame_count = 0

        while frame_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Append the raw frame to our in-memory list
            frames_list.append(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Extracted {frame_count}/{frames_to_extract} frames")

        cap.release()
        logger.info(f"Extracted {len(frames_list)} frames at {fps} FPS, dimensions: {frame_width}x{frame_height}")
        return frames_list, fps, frame_width, frame_height

    @staticmethod
    def make_video_from_frames(
        frames_list: List[np.ndarray], 
        output_video_path: str, 
        fps: float, 
        frame_width: int, 
        frame_height: int
    ) -> None:
        """Create a video from frames using a cloud-friendly codec."""
        if not frames_list:
            format_logger.warning("No frames found. Exiting without creating video.")
            return

        # Use H.264 codec (highly compatible) instead of mp4v
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # If CUDA is available, use hardware acceleration
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        else:
            # Fallback to H.264 without hardware acceleration
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        
        # Make sure dimensions are even (required by some codecs)
        frame_width = frame_width if frame_width % 2 == 0 else frame_width + 1
        frame_height = frame_height if frame_height % 2 == 0 else frame_height + 1
        
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            # If H.264 fails, try a more universal fallback
            format_logger.warning("Failed to open video writer with H.264, trying MPEG-4...")
            out = cv2.VideoWriter(
                output_video_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (frame_width, frame_height)
            )
        
        # Resize frames if needed to match the specified dimensions
        for frame in frames_list:
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                frame = cv2.resize(frame, (frame_width, frame_height))
            out.write(frame)
        
        out.release()
        
        # Validate the created file
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            format_logger.info(f"Successfully created video at {output_video_path}")
        else:
            format_logger.error(f"Failed to create valid video at {output_video_path}")

    @staticmethod
    def ensure_cloud_compatible_video(video_path: str, log_file_path: str = "format_validate.log") -> bool:
        """Post-process a video to ensure cloud compatibility using ffmpeg."""
        try:
            output_path = video_path.replace('.mp4', '_cloud.mp4')
            
            # Use ffmpeg to convert to a highly compatible format
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-c:v', 'libx264', '-preset', 'medium',
                '-profile:v', 'baseline', '-level', '3.0',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                output_path
            ]
            
            # Redirect stdout and stderr to our log file
            format_logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"\n--- FFMPEG CONVERSION: {video_path} ---\n")
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False
                )
                log_file.write(process.stdout)
                
            if process.returncode != 0:
                format_logger.error(f"FFmpeg error (code {process.returncode})")
                return False
                
            # Replace original with the compatible version
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                os.replace(output_path, video_path)
                format_logger.info(f"Video optimized for cloud compatibility: {video_path}")
                return True
            
            format_logger.error(f"Failed to create valid optimized video")
            return False
        except Exception as e:
            format_logger.exception(f"Error ensuring cloud compatibility: {e}")
            return False
