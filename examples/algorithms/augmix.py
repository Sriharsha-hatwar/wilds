# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 08:18:55 2021

@author: JeanMichelAmath
"""



import torch.nn.functional as F

import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to

class AugMix(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

    def objective(self, results):
        # In case we are in training mode i.e the data loader output tuples
        if results['y_pred'].size(0) != results['y_true'].size(0):
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                results['y_pred'], results['y_true'].size(0)) 
            # import pdb; pdb.set_trace()
            results['y_pred'] = logits_clean
            
            loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
            
            p_clean, p_aug1, p_aug2 = F.softmax(
              logits_clean, dim=1), F.softmax(
                  logits_aug1, dim=1), F.softmax(
                      logits_aug2, dim=1)
    
            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            js_div = 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            loss += js_div
            if torch.isnan(loss):
                import pdb; pdb.set_trace()
        else:
            loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        return loss
    
    def process_batch(self, batch):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor)
                - g (Tensor)
                - metadata (Tensor)
                - output (Tensor)
                - y_true
        """
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        # Here we concatenate only if we are in training mode. We do this here because, we don't have the information
        # about the training or validation mode within the AugMix algorithm
        # import pdb; pdb.set_trace()
        if isinstance(x, list):
            x_all = torch.cat(x, 0)
        else:
            x_all = x

        if self.model.needs_y:
            if self.training:
                outputs = self.model(x_all, y_true)
            else:
                outputs = self.model(x_all, None)
        else:
            outputs = self.model(x_all)
        
        if torch.isnan(outputs).sum() >= 1:
            import pdb; pdb.set_trace()
            
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            }
        return results
    
    