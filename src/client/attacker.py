import torch
import numpy as np
import copy
from torch.utils.data import TensorDataset
class Attacker:
    def __init__(self, config):
        """
        Args:
            config (dict): Loaded from configs/baseline.yaml OR configs/attack_backdoor.yaml
        """
        # 1. Handle Hydra config structure (cfg.attack) if it exists
        if hasattr(config, "attack"):
            self.config = config.attack
        else:
            self.config = config

        # 2. ROBUST KEY CHECK
        self.attack_type = self.config.get('type', self.config.get('attack_type', 'clean'))
        self.poison_ratio = self.config.get('poison_ratio', 0.0)

    def poison_dataset(self, dataset):
        """
        Takes a clean dataset, returns a poisoned TensorDataset.
        """
        if self.attack_type == 'clean' or self.poison_ratio <= 0:
            return dataset

        print(f"😈 Red Team: Executing '{self.attack_type}' attack...")

        X_local, y_local = self._extract_tensors(dataset)
        
        num_samples = len(X_local)
        num_poison = int(num_samples * self.poison_ratio)
        
        if num_poison == 0:
            return dataset
            
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)
        print(f"   -> Poisoning {num_poison}/{num_samples} samples.")

        if self.attack_type == 'backdoor':
            X_local, y_local = self._inject_backdoor(X_local, y_local, poison_indices)
            
        elif self.attack_type == 'label_flip':
            y_local = self._flip_labels(y_local, poison_indices)

        return TensorDataset(X_local, y_local)

    def _extract_tensors(self, dataset):
        if hasattr(dataset, 'indices'):
            X_list = []
            y_list = []
            for i in range(len(dataset)):
                x, y = dataset[i]
                X_list.append(x)
                y_list.append(y)
            return torch.stack(X_list).clone(), torch.tensor(y_list).clone()
            
        elif hasattr(dataset, 'tensors'):
            return dataset.tensors[0].clone(), dataset.tensors[1].clone()
            
        else:
            X_list = [dataset[i][0] for i in range(len(dataset))]
            y_list = [dataset[i][1] for i in range(len(dataset))]
            return torch.stack(X_list).clone(), torch.tensor(y_list).clone()

    def _inject_backdoor(self, X, y, indices):
        try:
            feat_idx = self.config.trigger_feat_idx
            trig_val = self.config.trigger_value
            target = self.config.target_label
        except AttributeError:
            feat_idx = self.config.get('trigger_feat_idx', 0)
            trig_val = self.config.get('trigger_value', 0.0)
            target = self.config.get('target_label', 0)

        X[indices, feat_idx] = trig_val
        y[indices] = target
        return X, y

    def _flip_labels(self, y, indices):
        try:
            source = self.config.source_label
            target = self.config.flip_to_label
        except AttributeError:
            source = self.config.get('source_label')
            target = self.config.get('flip_to_label')
        
        mask = (y[indices] == source)
        affected_indices = indices[mask]
        
        y[affected_indices] = target
        return y

    def scale_update(self, global_weights, local_weights):
        """
        Master function for post-training weight modification.
        Routes to either Stealth Scaling or Aggressive Boosting.
        """
        aggressive = self.config.get('aggressive', False)
        stealth = self.config.get('stealth', False)
        
        # 🥷 1. Handle Stealth Attack (Krum Bypass)
        if stealth:
            target_norm = self.config.get('target_norm_bound', 3.5)
            return self.apply_stealth_scaling(local_weights, global_weights, target_norm)
            
        # 😈 2. Handle Aggressive Attack (Model Replacement)
        if aggressive:
            scale_factor = self.config.get('estimated_n_clients', 1.0)
            if scale_factor > 1.0:
                print(f"😈 Red Team: Boosting weights by {scale_factor}x (Model Replacement)")
                scaled_weights = copy.deepcopy(local_weights)
                for key in global_weights.keys():
                    delta = local_weights[key] - global_weights[key]
                    scaled_weights[key] = global_weights[key] + (delta * scale_factor)
                return scaled_weights

        # 🤷 3. Standard Poisoning (No scaling)
        return local_weights
    
    def apply_stealth_scaling(self, w_malicious, w_global, target_norm_bound):
        """
        Shrinks the malicious update so it passes Krum's distance checks.
        """
        w_stealth = {}
        
        # 1. Calculate the raw update (delta)
        delta_w = {}
        current_norm = 0.0
        for key in w_malicious.keys():
            delta_w[key] = w_malicious[key] - w_global[key]
            current_norm += torch.norm(delta_w[key].float()) ** 2
        
        current_norm = torch.sqrt(current_norm)
        
        # 2. Check if we are too "loud" for Krum
        if current_norm > target_norm_bound:
            # We are too loud! Scale down to exactly the target bound.
            scale_factor = target_norm_bound / current_norm
            print(f"🥷 Stealth Attack: L2 Norm was {current_norm:.2f}. Scaling down by {scale_factor:.4f} to match bound ({target_norm_bound}).")
            
            for key in w_malicious.keys():
                w_stealth[key] = w_global[key] + (delta_w[key] * scale_factor)
        else:
            # We are quiet enough.
            print(f"🥷 Stealth Attack: L2 Norm ({current_norm:.2f}) is safely under bound ({target_norm_bound}). No scaling needed.")
            w_stealth = w_malicious
            
        return w_stealth
