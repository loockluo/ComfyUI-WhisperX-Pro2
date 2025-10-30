import os
import srt
import torch
import time
import whisperx
import folder_paths
import cuda_malloc
import translators as ts
from tqdm import tqdm
from datetime import timedelta
input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class PreViewSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)},
                }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    
    FUNCTION = "show_srt"

    def show_srt(self, srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r') as f:
            srt_content = f.read()
        return {"ui": {"srt":[srt_content,srt_name,dir_name]}}


class SRTToString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)},
                }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "read"

    CATEGORY = "AIFSH_FishSpeech"

    def read(self,srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r', encoding="utf-8") as f:
            srt_content = f.read()
        return (srt_content,)


class WhisperX:
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["large-v3","distil-large-v3","large-v2", "large-v3-turbo"]
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing',
        'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
        'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
        'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
        'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
        'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
        'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
        'yeekit', 'youdao']
        lang_list = ["zh","en","ja","ko","ru","fr","de","es","pt","it","ar"]
        return {"required":
                    {"audio": ("AUDIOPATH",),
                     "model_type":(model_list,{
                         "default": "large-v3"
                     }),
                     "batch_size":("INT",{
                         "default": 4
                     }),
                     "if_mutiple_speaker":("BOOLEAN",{
                         "default": False
                     }),
                     "use_auth_token":("STRING",{
                         "default": "put your huggingface user auth token here for Assign speaker labels"
                     }),
                     "if_translate":("BOOLEAN",{
                         "default": False
                     }),
                     "translator":(translator_list,{
                         "default": "alibaba"
                     }),
                     "to_language":(lang_list,{
                         "default": "en"
                     })
                     },
                }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ("SRT","SRT")
    RETURN_NAMES = ("ori_SRT","trans_SRT")
    FUNCTION = "get_srt"

    def get_srt(self, audio,model_type,batch_size,if_mutiple_speaker,
                use_auth_token,if_translate,translator,to_language):
        compute_type = "float16"

        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
        # 1. Transcribe with original whisper (batched)
        if model_type == "large-v3-turbo":
            model_type = "deepdml/faster-whisper-large-v3-turbo-ct2"
        model = whisperx.load_model(model_type, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio)
        result = model.transcribe(audio, batch_size=batch_size)
        # print(result["segments"]) # before alignment
        language_code=result["language"]
        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # print(result["segments"]) # after alignment
        
        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model_a,model
        if if_mutiple_speaker:
            # 3. Assign speaker labels
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)

            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio)
            # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            import gc; gc.collect(); torch.cuda.empty_cache(); del diarize_model
        # print(diarize_segments)
        # print(result.segments) # segments are now assigned speaker IDs
        
        srt_path = os.path.join(out_path,f"{time.time()}_{base_name}.srt")
        trans_srt_path = os.path.join(out_path,f"{time.time()}_{base_name}_{to_language}.srt")
        srt_line = []
        trans_srt_line = []
        for i, res in enumerate(tqdm(result["segments"],desc="Transcribing ...", total=len(result["segments"]))):
            start = timedelta(seconds=res['start'])
            end = timedelta(seconds=res['end'])
            try:
                speaker_name = res["speaker"][-1]
            except:
                speaker_name = "0"
            content = res['text']
            srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=speaker_name+content))
            if if_translate:
                #if i== 0:
                   # _ = ts.preaccelerate_and_speedtest() 
                content = ts.translate_text(query_text=content, translator=translator,to_language=to_language)
                trans_srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=speaker_name+content))
                
        with open(srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(srt_line))
        with open(trans_srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(trans_srt_line))

        if if_translate:
            return (srt_path,trans_srt_path)
        else:
            return (srt_path,srt_path)

class LoadAudioPath:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a", "mp4"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ("AUDIOPATH",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)


class WhisperXTextAlign:
    @classmethod
    def INPUT_TYPES(s):
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing',
        'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
        'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
        'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
        'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
        'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
        'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
        'yeekit', 'youdao']
        lang_list = ["zh","en","ja","ko","ru","fr","de","es","pt","it","ar","nl","uk","cs","pl","hu","fi","fa","el","tr","da","he","vi","te","hi","ca","ml","no","nn"]
        return {"required":
                    {"audio": ("AUDIOPATH",),
                     "text": ("STRING", {
                         "multiline": True,
                         "default": "在这里输入需要对齐的文本..."
                     }),
                     "language": (lang_list, {
                         "default": "zh"
                     }),
                     "if_mutiple_speaker": ("BOOLEAN", {
                         "default": False
                     }),
                     "use_auth_token": ("STRING", {
                         "default": "put your huggingface user auth token here for Assign speaker labels"
                     }),
                     "if_translate": ("BOOLEAN", {
                         "default": False
                     }),
                     "translator": (translator_list, {
                         "default": "alibaba"
                     }),
                     "to_language": (lang_list, {
                         "default": "en"
                     })
                     },
                }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ("SRT","SRT")
    RETURN_NAMES = ("ori_SRT","trans_SRT")
    FUNCTION = "align_text"

    def align_text(self, audio, text, language, if_mutiple_speaker,
                   use_auth_token, if_translate, translator, to_language):
        compute_type = "float16"

        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"

        # 1. Load audio
        audio_data = whisperx.load_audio(audio)

        # 2. Load VAD model to detect speech segments
        from whisperx.vad import load_vad_model, merge_chunks
        from whisperx.audio import SAMPLE_RATE
        vad_model = load_vad_model(device)
        vad_segments = vad_model({"waveform": torch.from_numpy(audio_data).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size=30,
            onset=0.500,
            offset=0.363,
        )

        if len(vad_segments) == 0:
            raise ValueError("No speech detected in audio file")

        # 3. Preprocess text: split into sentences
        import nltk
        try:
            # Try to use punkt tokenizer
            from nltk.tokenize import sent_tokenize
            # Download punkt if not available
            try:
                sentences = sent_tokenize(text, language='english' if language == 'en' else language)
            except:
                # Fallback: download punkt
                nltk.download('punkt', quiet=True)
                sentences = sent_tokenize(text, language='english' if language == 'en' else language)
        except:
            # Fallback: simple split by newlines or periods
            if '\n' in text:
                sentences = [s.strip() for s in text.split('\n') if s.strip()]
            else:
                sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

        # 4. Map sentences to VAD segments
        segments = []
        num_sentences = len(sentences)
        num_vad = len(vad_segments)

        if num_sentences == 0:
            raise ValueError("No sentences found in text")

        # Simple strategy: distribute sentences evenly across VAD segments
        if num_sentences <= num_vad:
            # More VAD segments than sentences, assign one sentence per VAD segment
            for i, sentence in enumerate(sentences):
                if i < len(vad_segments):
                    segments.append({
                        "text": sentence,
                        "start": vad_segments[i]["start"],
                        "end": vad_segments[i]["end"]
                    })
        else:
            # More sentences than VAD segments, group sentences
            sentences_per_vad = num_sentences / num_vad
            sentence_idx = 0
            for vad_idx, vad_seg in enumerate(vad_segments):
                # Calculate how many sentences for this VAD segment
                start_sentence_idx = sentence_idx
                end_sentence_idx = min(int((vad_idx + 1) * sentences_per_vad), num_sentences)

                # Combine sentences for this segment
                combined_text = " ".join(sentences[start_sentence_idx:end_sentence_idx])

                if combined_text.strip():
                    segments.append({
                        "text": combined_text,
                        "start": vad_seg["start"],
                        "end": vad_seg["end"]
                    })

                sentence_idx = end_sentence_idx

        # 5. Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

        # 6. Align text to audio
        result = whisperx.align(segments, model_a, metadata, audio_data, device, return_char_alignments=False)

        # Clean up alignment model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        del model_a

        # 7. Optional: Assign speaker labels
        if if_mutiple_speaker:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)
            diarize_segments = diarize_model(audio_data)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            gc.collect()
            torch.cuda.empty_cache()
            del diarize_model

        # 8. Generate SRT files
        srt_path = os.path.join(out_path, f"{time.time()}_{base_name}_aligned.srt")
        trans_srt_path = os.path.join(out_path, f"{time.time()}_{base_name}_aligned_{to_language}.srt")

        srt_line = []
        trans_srt_line = []

        for i, res in enumerate(tqdm(result["segments"], desc="Generating SRT...", total=len(result["segments"]))):
            start = timedelta(seconds=res['start'])
            end = timedelta(seconds=res['end'])

            # Try to get speaker name
            try:
                # Check if there are words with speaker info
                speaker_name = ""
                if 'words' in res and len(res['words']) > 0 and 'speaker' in res['words'][0]:
                    speaker_name = res['words'][0]["speaker"][-1] + ": "
            except:
                speaker_name = ""

            content = res['text']
            srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=speaker_name + content))

            if if_translate:
                translated_content = ts.translate_text(query_text=content, translator=translator, to_language=to_language)
                trans_srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=speaker_name + translated_content))

        with open(srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(srt_line))
        with open(trans_srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(trans_srt_line))

        if if_translate:
            return (srt_path, trans_srt_path)
        else:
            return (srt_path, srt_path)
