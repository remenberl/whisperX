import logging
import os
from typing import Callable, List, Union, Optional, NamedTuple

import ctranslate2
import faster_whisper
import numpy as np
from opencc import OpenCC
import torch
from tqdm.contrib import concurrent as tqdm_concurrent
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import PADDED_LEFT_SECOND, N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .alignment import find_alignments
from .quality import maybe_reject_reason
from .vad import load_vad_model, merge_chunks, merge_intervals

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens


def load_model(i18n_whisper_arch,
               en_whisper_arch,
               allowed_languages,
               device,
               device_index=0,
               compute_type="float16",
               asr_options=None,
               vad_options=None,
               download_root=None,
               threads=4):
    '''Load a Whisper model for inference.
    Args:
        whisper_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    '''


    i18n_model = WhisperModel(i18n_whisper_arch,
                              device=device,
                              device_index=device_index,
                              compute_type=compute_type,
                              download_root=download_root,
                              cpu_threads=threads)
    en_model = WhisperModel(en_whisper_arch,
                            device=device,
                            device_index=device_index,
                            compute_type=compute_type,
                            download_root=download_root,
                            cpu_threads=threads)
    tokenizer = None

    default_asr_options = {
        "beam_size": 3,
        "best_of": 3,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": "0",
        "hallucination_silence_threshold": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(
        **default_asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    vad_model = load_vad_model(torch.device(
        device), use_auth_token=None, **default_vad_options)

    return FasterWhisperPipeline(
        i18n_model=i18n_model,
        en_model=en_model,
        vad=vad_model,
        allowed_languages=allowed_languages,
        options=default_asr_options,
        tokenizer=tokenizer,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
        device="cuda:0"
    )


class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(self, encodings,
                                 tokenizer: faster_whisper.tokenizer.Tokenizer,
                                 options: faster_whisper.transcribe.TranscriptionOptions,
                                 **forward_params: dict):
        batch_size = encodings.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )


        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        encoder_output = ctranslate2.StorageView.from_array(encodings)
        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            length_penalty=options.length_penalty,
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
            max_length=self.max_length,
            return_scores=True,
            return_no_speech_prob=True,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
            max_initial_timestamp_index=max_initial_timestamp_index,
            #sampling_temperature=0.1,
            #beam_size=options.beam_size,
        )

        tokens_batch = [x.sequences_ids[0] for x in result]
       
        avg_logprobs = []
        no_speech_probs = []
        for i, tokens in enumerate(tokens_batch):
            seq_len = len(tokens)
            cum_logprob = result[i].scores[0] * seq_len
            avg_logprob = cum_logprob / (seq_len + 1)
            avg_logprobs.append(avg_logprob)
            no_speech_probs.append(result[i].no_speech_prob)
        def decode_batch(tokens: List[List[int]]) -> List[str]:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)
        lang = forward_params["language"]
        if lang.startswith("zh"):
            if lang == "zh" or lang == "zh_Hans":
                converter = forward_params["t2s"]
            else:
                converter = forward_params["s2t"]
            for i, t in enumerate(text):
                text[i] = converter.convert(t)
            tokens_batch = [encoding.ids for encoding in tokenizer.tokenizer.encode_batch(text)] 
        alignments = find_alignments(self.model, tokenizer, tokens_batch, encoder_output)
        return text, alignments, avg_logprobs, no_speech_probs

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(
            self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)


class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
            self,
            i18n_model,
            en_model,
            vad,
            allowed_languages: list[str],
            vad_params: dict,
            options: NamedTuple,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework="pt",
            language: Optional[str] = None,
            suppress_numerals: bool = False,
            **kwargs
    ):
        self.i18n_model = i18n_model
        self.en_model = en_model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params
        self.t2s = OpenCC('t2s')
        self.s2t = OpenCC('s2t')
        self.allowed_languages = set(allowed_languages)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        forward_kwargs = {}
        if "language" in kwargs:
            forward_kwargs["language"] = kwargs["language"]
        if "t2s" in kwargs:
            forward_kwargs["t2s"] = kwargs["t2s"]
        if "s2t" in kwargs:
            forward_kwargs["s2t"] = kwargs["s2t"]
        return preprocess_kwargs, forward_kwargs, {}

    def preprocess(self, inputs):
        return {'inputs': inputs['inputs']}

    def _forward(self, model_inputs, **forward_parameters: dict):
        outputs = self.model.generate_segment_batched(
            model_inputs['inputs'], self.tokenizer, self.options, **forward_parameters)
        return {'text': outputs[0], 'alignment': outputs[1], 'logprob': outputs[2], 'no_speech_prob': outputs[3]}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            stacked_encodings = torch.cat([x['inputs'] for x in items], 0)
            return {'inputs': stacked_encodings} 
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(
            dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(
            model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def detect_segment_language(self, audio: np.ndarray, seg, cache_encoding_device):
        f1 = int(seg['start'] * SAMPLE_RATE)
        f2 = int(seg['end'] * SAMPLE_RATE)
        languages = self.detect_language(audio[f1:f2], seg, cache_encoding_device)
        if len(languages) == 1:
            return [seg]
        segs_to_return = []
        segs_to_split = [seg]
        while segs_to_split:
            new_segs_to_split = []
            for seg in segs_to_split:
                chunk_size = (seg["end"] - seg["start"]) / 2
                subsegments = merge_intervals(seg["segments"], seg["weights"],
                                              chunk_size=chunk_size,
                                              min_drop_duration=0.1)
                if len(subsegments) == 1:
                    segs_to_return.append(seg)
                    continue
                prev_lang = seg["language"]
                for subseg in subsegments:
                    if subseg["end"] - subseg["start"] <= 2 and prev_lang:
                        languages = prev_lang
                        subseg["language"] = languages
                    else:
                        f1 = int(subseg["start"] * SAMPLE_RATE)
                        f2 = int(subseg["end"] * SAMPLE_RATE)
                        languages = self.detect_language(audio[f1:f2], subseg, cache_encoding_device, cache=False)
                    if len(languages) == 1 or subseg["end"]-subseg["start"] <= 2:
                        segs_to_return.append(subseg)
                    else:
                        new_segs_to_split.append(subseg)
                    prev_lang = languages
            segs_to_split = new_segs_to_split
        segs_to_return.sort(key=lambda seg: seg["start"])
        merged_segments = []
        prev_lang = ""
        for seg in segs_to_return:
            if seg["language"][0] != prev_lang:
                merged_segments.append(seg)
                prev_lang = seg["language"][0]
            else:
                merged_segments[-1]["end"] = max(merged_segments[-1]["end"], seg["end"])
                merged_segments[-1]["segments"] += seg["segments"]
                merged_segments[-1]["weights"] += seg["weights"]
        for seg in merged_segments:
            f1 = int(seg["start"] * SAMPLE_RATE)
            f2 = int(seg["end"] * SAMPLE_RATE)
            self.cache_encoding(audio[f1:f2], seg, seg["language"], cache_encoding_device)
        return merged_segments 


    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_threads=16,
        num_workers=0, language=None, task=None, chunk_size=30,
        print_progress=False, combined_progress=False,
        get_prompt_per_lang: Optional[Callable[[str], str]] = None,
        lang_reference="en", cache_encoding_device = "cuda",
    ) -> list:
        if isinstance(audio, str):
            audio = load_audio(audio)

        vad_segments = self.vad_model({"waveform": torch.from_numpy(
            audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size - PADDED_LEFT_SECOND,
            min_drop_duration=0.1,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        vad_segments_per_lang = {}
        if self.tokenizer:
            vad_segments_per_lang[language] = vad_segments
        else:
            logging.info("\tDetecting language for segments")
            results = tqdm_concurrent.thread_map(self.detect_segment_language,
                                                 [audio]*len(vad_segments),
                                                 vad_segments,
                                                 [cache_encoding_device]*len(vad_segments),
                                                 max_workers=num_threads,
                                                 disable=logging.root.level >= logging.INFO)
            for segments in results:
                for seg in segments:
                    lang = seg["language"][0]
                    if lang in vad_segments_per_lang:
                        vad_segments_per_lang[lang].append(seg)
                    else:
                        vad_segments_per_lang[lang] = [seg]
        results = []
        for lang, vad_segments in vad_segments_per_lang.items():
            logging.info(f"\tGetting prompts for {lang}")
            if lang == "zh" and lang_reference.startswith(lang):
                lang = lang_reference
            if get_prompt_per_lang:
                prompt = get_prompt_per_lang(lang)
                logging.info(f"\tPrompt: {prompt}")
                new_asr_options = self.options._asdict()
                new_asr_options["initial_prompt"] = prompt
                new_options = faster_whisper.transcribe.TranscriptionOptions(
                    **new_asr_options)
                self.options = new_options
            logging.info(f"\tTranscribing for {lang}")
            results.append(
                self.transcribe_per_lang(audio, vad_segments, batch_size,
                                         num_workers, lang, task,
                                         print_progress, combined_progress,
                                         cache_encoding_device))
        return results

    def transcribe_per_lang_with_split(
        self, audio, vad_segments, batch_size=None,
        num_workers=0, language='en', task=None, print_progress=False,
        combined_progress=False, min_drop_duration=0.1, cache_encoding_device="cuda"
    ):
        split_segments = []
        to_orig_map = {}
        for i, seg in enumerate(vad_segments):
            chunk_size = (seg["end"] - seg["start"]) / 2
            subsegments = merge_intervals(seg["segments"], seg["weights"], chunk_size, min_drop_duration)
            for subsegment in subsegments:
                f1 = int(subsegment["start"] * SAMPLE_RATE)
                f2 = int(subsegment["end"] * SAMPLE_RATE)
                self.cache_encoding(audio[f1:f2], subsegment, language, cache_encoding_device)
                to_orig_map[len(split_segments)] = i
                split_segments.append(subsegment)
        results = self.transcribe_per_lang(audio, split_segments, batch_size,
                                           num_workers, language, task,
                                           print_progress, combined_progress,
                                           cache_encoding_device)
        output_segments = [[] for _ in range(len(vad_segments))]
        for i, seg in enumerate(results["segments"]):
            output_segments[to_orig_map[i]].append(seg)
        return {"segments": output_segments, "language": results["language"]}

    def transcribe_per_lang(
        self, audio, vad_segments, batch_size=None,
        num_workers=0, language='en', task=None, print_progress=False,
        combined_progress=False, cache_encoding_device="cuda",
    ):
        def data(segments):
            for seg in segments:
                generate_cache = False
                if "encoding" not in seg:
                    generate_cache = True
                # For language mismatch, regenerates encoding.
                elif seg["encoding_lang"] != language and (
                        language == "en" or seg["encoding_lang"] == "en"):
                    generate_cache = True
                if generate_cache:
                    f1 = int(seg["start"] * SAMPLE_RATE)
                    f2 = int(seg["end"] * SAMPLE_RATE)
                    self.cache_encoding(audio[f1:f2], seg, language,
                                        cache_encoding_device)
                yield {'inputs': seg["encoding"]}

        tokenizer_lang = "zh" if language.startswith("zh") else language
        if self.tokenizer is None:
            task = task or "transcribe"
            self.model = self.en_model if language == "en" else self.i18n_model
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                self.model.model.is_multilingual, task=task,
                                                                language=tokenizer_lang)
        else:
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or tokenizer_lang != self.tokenizer.language_code:
                self.model = self.en_model if language == "en" else self.i18n_model
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                    self.model.model.is_multilingual, task=task,
                                                                    language=tokenizer_lang)

        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            # print(f"Suppressing numeral and symbol tokens: {numeral_symbol_tokens}")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = self.options._replace(
                suppress_tokens=new_suppressed_tokens)

        segments = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(
            language=language, t2s=self.t2s, s2t=self.s2t)
        for idx, out in enumerate(self.__call__(data(vad_segments), batch_size=batch_size,
                                                num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            alignment = out['alignment']
            if batch_size in [0, 1, None]:
                text = text[0]
                alignment = alignment[0]
            vad_start = round(vad_segments[idx]['start'], 3)
            vad_end = round(vad_segments[idx]['end'], 3)
            active_duration = 0
            for word in alignment:
                word["start"] -= PADDED_LEFT_SECOND
                word["start"] = max(0, word["start"])
                word["start"] += vad_start
                word["end"] -= PADDED_LEFT_SECOND
                word["end"] = max(0, word["end"])
                word["end"] += vad_start
                word["start"] = round(word["start"], 3)
                word["end"] = round(word["end"], 3)
                del word["probability"]
                del word["tokens"]
                if word["start"] < vad_start: 
                    continue
                if word["end"] > vad_end: 
                    continue
                word_duration = word["end"] - word["start"]
                active_duration += word_duration
            seg = {
                "text": text,
                "start": vad_start,
                "end": vad_end,
                "active_duration": round(active_duration, 3),
                "alignment": alignment,
                "segments": vad_segments[idx]["segments"],
                "weights": vad_segments[idx]["weights"],
                "logprob": round(out["logprob"], 3),
                "no_speech_prob": round(out["no_speech_prob"], 3),
                "language": tokenizer_lang,
            }
            seg["reason"] = maybe_reject_reason(seg)
            # Because seg has problem, encoding is returned for later reprocess.
            if seg["reason"]:
                seg["encoding"] = vad_segments[idx]["encoding"]
                seg["encoding_lang"] = vad_segments[idx]["encoding_lang"]
            segments.append(seg)

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = self.options._replace(
                suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": tokenizer_lang}

    def cache_encoding(self, audio: np.ndarray, seg, language=None, device="cuda"):
        feature = log_mel_spectrogram(audio[: N_SAMPLES], n_mels=80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        lang = language if language else seg["language"][0]
        if lang == "en":
            encoder_output = self.en_model.encode(feature)
        else:
            encoder_output = self.i18n_model.encode(feature)
        seg["encoding"] = torch.as_tensor(encoder_output, device=device)
        seg["encoding_lang"] = lang 

    def detect_language(self, audio: np.ndarray, seg, cache_encoding_device, cache=True):
        feature = log_mel_spectrogram(audio[: N_SAMPLES], n_mels=80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.i18n_model.encode(feature)
        results = self.i18n_model.model.detect_language(encoder_output)
        detected_languages = []
        probs = []
        for lang_token, prob in results[0]:
            lang = lang_token[2:-2]
            if prob < 0.01 and detected_languages:
                break
            if lang not in self.allowed_languages:
                continue
            if not detected_languages:
                detected_languages.append(lang)
                probs.append(prob)
            else:
                detected_languages.append(lang)
                probs.append(prob)
                break
        if not detected_languages:
            detected_languages.append("en")
            probs.append(1.0)
        assert 0 < len(detected_languages) <= 2
        # Swaps order
        if len(detected_languages) == 2 and detected_languages[0] == "en" and probs[1] >= 0.3:
            detected_languages = [detected_languages[1], "en"]
        if seg:
            seg["language"] = detected_languages
            if detected_languages[0] == "en":
                encoder_output = self.en_model.encode(feature)
            if cache:
                seg["encoding"] = torch.as_tensor(encoder_output, device=cache_encoding_device)
                seg["encoding_lang"] = detected_languages[0]
        return detected_languages 
        # In bilingual setting, prefers non-English output.
        if first_language == "en" and second_language_probability > 0.4:
            return second_language
        # print(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
        return first_language
