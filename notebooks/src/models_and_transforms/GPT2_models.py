from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, GPT2Model
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.optimization import AdamW

from transformers.file_utils import ModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

class GPT2_Actor_Critic_Model(LightningModule):
    def __init__(self, config, pad_id, bos_id, **kwargs):
        super().__init__()
        self.config = config
        self.pad_token_id = pad_id
        self.bos_token_id = bos_id
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.value_head = nn.Linear(config.n_embd, 1, bias=False)
    
    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        amount_pad = (x == self.pad_token_id).sum(dim=1)
        input_ids = torch.cat([torch.zeros(batch_size,1, dtype=torch.long, device=self.device), x], dim=1)
        input_ids[torch.arange(batch_size), amount_pad] = self.bos_token_id
        
        attention_mask = (input_ids != self.pad_token_id)
        position_ids = torch.cumsum(attention_mask, dim=1)
        
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        value_togits = self.value_head(hidden_states)
        return lm_logits, value_togits
    
    def predict(self, x, **kwargs):
        x = x.to(self.device)
        policy_logits, value_logits = self.forward(x, **kwargs)
        policy_logit, value_logit = policy_logits[:,-1], value_logits[:, -1]
        p = torch.softmax(policy_logit, dim=-1)
        return p.data.cpu(),  value_logit.data.cpu()
    
    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        policy_logits, value_logits = self(x)
        policy_logit, value_logit = policy_logits[:,-1], value_logits[:, -1]
        loss = 0
        if 'target_value' in batch:
            target_value = batch["target_value"]
            value_loss = nn.MSELoss()(value_logit.view(-1), target_value.view(-1))
        else:
            value_loss = 0
        
        if 'target_policy' in batch:
            target_policy = batch["target_policy"]
            if target_policy.shape[1] == 1:
                # single word target
#                 print(policy_logit.view(-1,self.BERT.config.vocab_size))
#                 print(target_policy.view(-1))
                policy_loss = nn.CrossEntropyLoss()(policy_logit.view(-1,self.config.vocab_size), target_policy.view(-1))
            elif target_policy.shape[1] == self.config.vocab_size:
                # full policy target
                policy_loss = -torch.dot(target_policy.view(-1), torch.log(policy_logit.softmax(-1)).view(-1))/policy_logit.shape[0]
            else:
                raise f"incorrect shape for target_policy {target_policy.shape}"
        else:
            target_policy = 0
            
        loss = value_loss + policy_loss
            
        return {"loss":loss, 'log': {'train_loss': loss}}
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=10000, epochs=1)
#         scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=15000)
        return optimizer#, [scheduler]
    

class GPT2_Scratch_Pad_Model(GPT2LMHeadModel, LightningModule):
    def training_step(self, batch, batch_idx, **model_kwargs):
        input_ids = batch["input_ids"]
        target_ids = batch['target_ids']
        pad_id = batch['pad_id']
        model_kwargs['attention_mask'] = batch['attention_mask']
        
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs)
        logits = outputs[0]
                
        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss = loss_fct(logits.view(-1, self.config.vocab_size), target_ids.view(-1))
        if torch.isnan(loss):
            print(f'input_ids is nan:{torch.isnan(batch["input_ids"])}, decoder_input_ids is nan:{torch.isnan(batch["decoder_input_ids"])}')
            print(f'logits={logits}')
            
        return {"loss":loss, 'logits':logits}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        return [optimizer], [scheduler]
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        postfix_additional_tokens_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        # set init values
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if input_ids is None:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id, **model_kwargs
            )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )
                    
        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                postfix_additional_tokens_fn=postfix_additional_tokens_fn,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            batch_size = input_ids.shape[0] * num_return_sequences

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # interleave with `num_beams * num_return_sequences`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            print('foo')

            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                postfix_additional_tokens_fn=postfix_additional_tokens_fn,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            diverse_beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            return self.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        postfix_additional_tokens_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **model_kwargs
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        
        print("i'm in beam search mode")

        while cur_len < max_length:
            input_ids = postfix_additional_tokens_fn(input_ids)
            model_kwargs['attention_mask'] = (input_ids!=pad_token_id).to(torch.int)
            
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # adjust tokens for Bart, *e.g.*
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = input_ids.shape[1]

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break
        
        beam_scorer.max_length = cur_len
        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return decoded
    
    
    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        postfix_additional_tokens_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **model_kwargs
    ):
        print("i'm in my beam sample")

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while cur_len < max_length:
#             print("pre postfix_additional_tokens_fn")
#             print('input_ids', input_ids)
#             print('model_kwargs', model_kwargs)
            input_ids = postfix_additional_tokens_fn(input_ids)
            model_kwargs['attention_mask'] = (input_ids!=pad_token_id).to(torch.int)
#             print("post postfix_additional_tokens_fn")
#             print('input_ids', input_ids)
#             print('model_kwargs', model_kwargs)
            
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
#             print('model_inputs: ', model_inputs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # adjust token scores (a no-op by default)
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
#             print("prior to cat")
#             print("input_ids", input_ids)
#             print("beam_next_tokens", beam_next_tokens)

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = input_ids.shape[1]
            
#             print("post cat")
#             print("input_ids", input_ids)
#             print("beam_next_tokens", beam_next_tokens)
#             print("input_ids pre postfix_additional_tokens_fn")
#             print(input_ids)
#             input_ids = postfix_additional_tokens_fn(input_ids)
#             print("input_ids post postfix_additional_tokens_fn")
#             print(input_ids)
#             cur_len = input_ids.shape[1]
#             model_kwargs['attention_mask'] = (input_ids!=pad_token_id).to(torch.int)[:,:-1]
            
            # This is where I need to update model_kwargs['attention_mask'] to include the latest added tokens
            # the rest should follow nicely
            # However, there is a little bit of weird behaviour with the last token duplication,
            # I might want to make surrer I understand that before I move on.
            
#             print("doing it again")
#             print(input_ids)
#             print(cur_len)
#             print("pre _update_model_kwargs_for_generation")
#             print(model_kwargs)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
#             print("post _update_model_kwargs_for_generation")
#             print(model_kwargs)
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break
                
                
        beam_scorer.max_length = cur_len
        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return decoded