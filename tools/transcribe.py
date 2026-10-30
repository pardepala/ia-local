#!/usr/bin/env python3
"""
Script de transcripción de audio y video usando faster-whisper con aceleración GPU (CUDA).
Soporta archivos: MP3, MP4, WAV, MKV, M4A, FLAC, OGG, WEBM, etc.
Genera salidas en:
  - .txt (texto corrido completo)
  - .srt (subtítulos estándar con timestamps)
  - .json (metadatos detallados y segmentos)
"""

import os
import sys

# Asegurar codificación UTF-8 en consola de Windows y cargar DLLs de CUDA
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    
    # Cargar rutas de DLLs de NVIDIA (cublas, cudnn) instaladas vía pip
    try:
        import site
        from pathlib import Path
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            nvidia_base = Path(site_dir) / "nvidia"
            if nvidia_base.exists():
                for bin_dir in nvidia_base.glob("*/bin"):
                    if bin_dir.is_dir():
                        os.add_dll_directory(str(bin_dir))
                        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

# Desactivar advertencia estética de enlaces simbólicos en Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import json
import argparse
import io
import gc
from pathlib import Path
import numpy as np
import av
import faster_whisper.audio
from faster_whisper import WhisperModel
import ctranslate2

# Parche de compatibilidad PyAV 19+ (omisión del parámetro deprecado 'metadata_errors')
def _safe_decode_audio(input_file, sampling_rate: int = 16000, split_stereo: bool = False):
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )
    raw_buffer = io.BytesIO()
    dtype = None
    try:
        container = av.open(input_file, mode="r", metadata_errors="ignore")
    except TypeError:
        container = av.open(input_file, mode="r")
        
    with container:
        frames = container.decode(audio=0)
        frames = faster_whisper.audio._ignore_invalid_frames(frames)
        frames = faster_whisper.audio._group_frames(frames, 500000)
        frames = faster_whisper.audio._resample_frames(frames, resampler)
        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)
            
    del resampler
    gc.collect()
    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
    audio = audio.astype(np.float32) / 32768.0
    if split_stereo:
        audio = audio.reshape((-1, 2)).T
    return audio

faster_whisper.audio.decode_audio = _safe_decode_audio

# Formatos multimedia admitidos comúnmente
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"
}

def format_timestamp_srt(seconds: float) -> str:
    """Convierte segundos a formato SRT (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_file(
    file_path: Path,
    model: WhisperModel,
    output_dir: Path,
    language: str | None = None,
    task: str = "transcribe"
):
    print(f"\n[+] Procesando: {file_path.name}")
    start_time = time.time()
    
    # Inferencia con Whisper
    segments, info = model.transcribe(
        str(file_path),
        beam_size=5,
        language=language,
        task=task,
        vad_filter=True, # Filtra silencios automáticamente
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    detected_lang = info.language
    prob = info.language_probability
    duration = info.duration
    print(f"    - Idioma detectado: {detected_lang} (confianza: {prob*100:.1f}%)")
    print(f"    - Duración del archivo: {duration:.1f} segundos ({duration/60:.2f} min)")
    print("    - Transcribiendo segmentos...")
    
    collected_segments = []
    full_text_parts = []
    srt_lines = []
    
    for idx, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        full_text_parts.append(text)
        
        start_srt = format_timestamp_srt(segment.start)
        end_srt = format_timestamp_srt(segment.end)
        srt_lines.append(f"{idx}\n{start_srt} --> {end_srt}\n{text}\n")
        
        collected_segments.append({
            "id": idx,
            "start": segment.start,
            "end": segment.end,
            "text": text
        })
        
        # Feedback en tiempo real para grabaciones largas
        print(f"      [{start_srt} -> {end_srt}] {text}")
        
    elapsed = time.time() - start_time
    speed_factor = duration / elapsed if elapsed > 0 and duration > 0 else 0
    print(f"    ✔ Finalizado en {elapsed:.2f}s ({speed_factor:.1f}x más rápido que tiempo real)")
    
    # Asegurar carpeta de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = file_path.stem
    
    # 1. Guardar TXT
    txt_path = output_dir / f"{base_name}.txt"
    txt_content = " ".join(full_text_parts)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"    📄 Texto guardado en: {txt_path}")
    
    # 2. Guardar SRT
    srt_path = output_dir / f"{base_name}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    print(f"    🎬 Subtítulos guardados en: {srt_path}")
    
    # 3. Guardar JSON
    json_path = output_dir / f"{base_name}.json"
    result_data = {
        "file": file_path.name,
        "language": detected_lang,
        "language_probability": prob,
        "duration": duration,
        "elapsed_seconds": elapsed,
        "segments": collected_segments,
        "text": txt_content
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"    📦 Datos completos en: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Transcriptor local de audio y video con faster-whisper")
    parser.add_argument(
        "input",
        nargs="?",
        default="data/input",
        help="Ruta al archivo multimedia o directorio (por defecto: 'data/input/')"
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        help="Modelo Whisper a utilizar (por defecto: 'large-v3-turbo')"
    )
    parser.add_argument(
        "--output",
        default="data/output",
        help="Carpeta de destino para las transcripciones (por defecto: 'data/output/')"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Código de idioma (ej. 'es', 'en'). Si no se indica, se detecta automáticamente."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Dispositivo de procesamiento (por defecto detecta CUDA automáticamente)"
    )

    args = parser.parse_args()
    
    # Detección de dispositivo
    has_cuda = ctranslate2.get_cuda_device_count() > 0
    if args.device == "auto":
        device = "cuda" if has_cuda else "cpu"
    else:
        device = args.device
        
    compute_type = "float16" if device == "cuda" else "int8"
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        files_to_process = [p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not files_to_process:
            print(f"[!] No se encontraron archivos de audio/video en '{input_path}'.")
            print(f"    Coloca archivos soportados ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) en la carpeta.")
            return
    else:
        print(f"[!] Error: La ruta '{input_path}' no existe.")
        print(f"    Uso: .\\transcribir.bat \"ruta\\a\\tu_video.mp4\"")
        sys.exit(1)

    print(f"==================================================")
    print(f"  🎙️ Transcriptor Local de Audio/Video")
    print(f"  Dispositivo: {device.upper()} (Compute: {compute_type})")
    print(f"  Modelo:      {args.model}")
    print(f"  Archivos:    {len(files_to_process)} archivo(s)")
    print(f"==================================================")
    
    print(f"[*] Cargando modelo '{args.model}' (se descargará automáticamente la primera vez)...")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    print("[*] Modelo cargado con éxito.")
    
    for f in files_to_process:
        transcribe_file(f, model, output_dir, language=args.language)

if __name__ == "__main__":
    main()
