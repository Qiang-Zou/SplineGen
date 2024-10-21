from torch import nn
import torch
from .onn import TOKENS

class PointsEncoderDecoder(nn.Module):
    def __init__(self, encoder,decoder,lock_encoder=False) -> None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.lock_encoder=lock_encoder

    def forward(self,src,src_mask,tgt,tgt_mask):
        if self.lock_encoder:
            with torch.no_grad():
                encoder_output=self.encoder(src,src_mask)
        else:
            encoder_output=self.encoder(src,src_mask)

        decoder_output=self.decoder(encoder_output,src_mask,tgt,tgt_mask)
        return decoder_output

class PointsEncoderDecoder2(nn.Module):
    def __init__(self, encoder,decoder1,decoder2,masking_rate=None) -> None:
        super().__init__()
        self.encoder=encoder
        self.decoder1=decoder1
        self.decoder2=decoder2
        self.masking_rate=masking_rate
        if self.masking_rate:
            self.dropout=nn.Dropout(masking_rate)

    def forward(self,src,src_mask,tgt1,tgt1_mask,tgt2,tgt2_mask):
        encoder_output=self.encoder(src,src_mask)
        if self.masking_rate:
            src_mask=self.dropout(src_mask.float()).bool()
        decoder_output1=self.decoder1(encoder_output,src_mask,tgt1,tgt1_mask)
        decoder_output2=self.decoder2(encoder_output,src_mask,tgt2,tgt2_mask)
        return decoder_output1,decoder_output2

def pad_item_torch(
    points: torch.Tensor,
):
    n_tokens = len(TOKENS)
    
    # points_padded = np.zeros((self.max_samples + n_tokens, 3 + n_tokens),
    batch_size,max_len,dimension=points.shape
    points_padded = torch.zeros(
    (batch_size,max_len + n_tokens, dimension + n_tokens),
    dtype=torch.float32,device=points.device)

    points_padded[:,TOKENS['<eos>'], dimension] = 1.0
    # points_padded[n_tokens:n_tokens + len_points, :self.dimension] =\
    #   points[:len_points]

    points_padded[:,n_tokens:,:dimension]=points

    return points_padded

class PointsEncoderDecoder11(nn.Module):
    def __init__(self, encoder,encoder2,knot_decoder,param_decoder,param_decoder2,lock_encoder=False,lock_knots=False,disturb_knot=None) -> None:
        super().__init__()
        self.encoder=encoder
        self.encoder2=encoder2
        self.decoder_k=knot_decoder
        self.decoder_p=param_decoder
        self.decoder_p2=param_decoder2
        self.lock_encoder=lock_encoder
        self.lock_knots=lock_knots
        self.disturb_knot=disturb_knot

    def forward(self,points,params,points_mask,points_len,label,knots,knots_mask,eval=False,half_eval=False):
        if eval:
            return self.do_eval(points,points_mask,points_len)
        if half_eval:
            return self.half_eval(points,points_mask,points_len,label)
        if self.lock_encoder:
            with torch.no_grad():
                encoder_output=self.encoder(points,points_mask)
        else:
            encoder_output=self.encoder(points,points_mask)

        if self.lock_knots:
            with torch.no_grad():
                knots,knots_tmp,kv=self.decoder_k(encoder_output,points_mask,knots,knots_mask)
        else:
            knots,knots_tmp,kv=self.decoder_k(encoder_output,points_mask,knots,knots_mask)

        encoder_output2=self.encoder2(points,points_mask)
        embedding_expanded=pad_item_torch(encoder_output2)
        pointer_log_scores,pointer_argmax,decoder_output=self.decoder_p(embedding_expanded,encoder_output,points_mask,points_len,label,knots_embedding=knots_tmp,
                                                                        knots_kv=kv,
                                                                        knots_mask=knots_mask)
        params=self.decoder_p2(decoder_output,points_mask).squeeze()
        return knots,knots_mask,pointer_log_scores,pointer_argmax,params

    def do_eval(self,points,points_mask,points_len):
        with torch.no_grad():
            encoder_output=self.encoder(points,points_mask)
            knots,knots_mask,knots_internal,kv=self.gen_knots(encoder_output,points_mask)
            # embedding_expanded=pad_item_torch(encoder_output)
            encoder_output2=self.encoder2(points,points_mask)
            embedding_expanded=pad_item_torch(encoder_output2)
            pointer_log_scores,pointer_argmax,decoder_output=self.decoder_p(embedding_expanded,encoder_output,points_mask,points_len,None,knots_embedding=knots_internal,
                                                                            knots_mask=knots_mask,knots_kv=kv)
            params=self.decoder_p2(decoder_output,points_mask).squeeze()
            return knots,knots_mask,pointer_log_scores,pointer_argmax,params.clip(0,1)

    def half_eval(self,points,points_mask,points_len,label):
        with torch.no_grad():
            encoder_output=self.encoder(points,points_mask)
            knots,knots_mask,knots_internal,kv=self.gen_knots(encoder_output,points_mask)

        encoder_output2=self.encoder2(points,points_mask)
        embedding_expanded=pad_item_torch(encoder_output2)
        pointer_log_scores,pointer_argmax,decoder_output=self.decoder_p(embedding_expanded,encoder_output,points_mask,points_len,label,
                                                                        knots_embedding=knots_internal,
                                                                        knots_kv=kv,
                                                                        knots_mask=knots_mask)
        # pointer_log_scores,pointer_argmax,decoder_output=self.decoder_p(embedding_expanded,points_len,None,knots_embedding=knots_internal,knots_mask=knots_mask)
        params=self.decoder_p2(decoder_output,points_mask).squeeze()
        return knots,knots_mask,pointer_log_scores,pointer_argmax,params.clip(0,1)

    def gen_knots(self,embedding,embedding_mask,p=3):
        src=embedding
        src_mask_position=embedding_mask
        device = src.device

        batch_size, max_seq_len, _ = src.size()
        
        # Assuming a predefined max_knots_len
        max_knots_len = 30 - 1  # You can adjust this value as needed
        
        with torch.no_grad():
            # Initialize the tgt tensor with zeros and the first column as SOS
            tgt = torch.zeros(batch_size, max_knots_len, 3, device=device)
            tgt[:, 0, 1] = 1  # Assuming the second channel is for SOS
            
            # Initialize the tgt_mask_position tensor with the first position as True
            tgt_mask_position = torch.zeros(batch_size, max_knots_len, dtype=torch.bool, device=device)
            tgt_mask_position[:, 0] = True

            # mask = torch.zeros(batch_size, max_knots_len, dtype=torch.bool, device=device)
            
            eos_detected = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
            # Threshold for detecting EOS
            eos_threshold = 0.5  # Adjust this value as needed
            
            if self.disturb_knot is not None:
                for i in range(1, max_knots_len):
                    # Predict the next knot using the KPN model
                    train_output,internal_output,kv = self.decoder_k(src, src_mask_position, tgt, tgt_mask_position)
                    eos_detected |= (train_output[:, i-1, 2] > eos_threshold)    
                    # Update the tgt tensor with the predicted value for the next iteration
                    if i > p+1:
                        tgt[:, i,1:] = train_output[:, i-1,1:]
                        tgt[:, i,0] = torch.where(eos_detected,train_output[:, i-1,0],
                                                  torch.clip(train_output[:, i-1,0]+torch.normal(mean=0.,std=self.disturb_knot,size=(batch_size,),device=device),1e-6,1-1e-6))
                    else:
                        tgt[:, i] = train_output[:, i-1]

                    # Update the tgt_mask_position tensor for the next iteration
                    tgt_mask_position[:, i] = torch.where(eos_detected, False, True)
            else:
                for i in range(1, max_knots_len):
                    # Predict the next knot using the KPN model
                    train_output,internal_output,kv = self.decoder_k(src, src_mask_position, tgt, tgt_mask_position)
                    eos_detected |= (train_output[:, i-1, 2] > eos_threshold)    
                    # Update the tgt tensor with the predicted value for the next iteration
                    tgt[:, i] = train_output[:, i-1]

                    # Update the tgt_mask_position tensor for the next iteration
                    tgt_mask_position[:, i] = torch.where(eos_detected, False, True)

            # index=torch.ones(batch_size, max_knots_len, dtype=torch.long, device=device)*max_knots_len
            index=torch.arange(max_knots_len, dtype=torch.long, device=device)+p+2
            index=index.clip(max=max_knots_len-1)
            mask=torch.gather(tgt_mask_position, 1, index.unsqueeze(0).expand(batch_size,-1))
            mask[:,-(p+1):]=False
            
        train_output,internal_output,kv = self.decoder_k(src, src_mask_position, tgt, tgt_mask_position)
        output=torch.zeros_like(train_output[...,0])
        output[:,p+1:]=train_output[:,p+1:,0]

        output=output.masked_fill_(~mask,1)
        # train_output=torch.masked_fill(output, ~mask, 1)
        train_output=torch.sort(output,dim=-1)[0]
        return train_output,tgt_mask_position,internal_output,kv

class SplineGen(nn.Module):
    def __init__(self, base_model,additional_model) -> None:
        super().__init__()
        self.base_model=base_model
        self.additional_model=additional_model

    def forward(self,points,params,points_mask,points_len,label,knots,knots_mask,eval=False,half_eval=True):
        with torch.no_grad():
            knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = self.base_model(
                points,params,points_mask, points_len,
                label,knots[:,:-1],knots_mask[:,:-1],half_eval=half_eval,eval=eval)

        if eval:
            indices=pointer_argmaxs.unsqueeze(-1).expand(-1, -1, 3).clip(0,points.size(1)-1).long()
            points=torch.gather(points,1,torch.where(indices==0,0,(indices-1)))

        params,knots=self.additional_model(points,params,points_mask,knots,knots_mask)

        return knots,knots_mask,log_pointer_scores, pointer_argmaxs,params