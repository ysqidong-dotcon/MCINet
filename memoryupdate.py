import torch


class MemoryManager:
    def __init__(self, cfg):
        self.cfg = cfg

        # Memory initialization
        self.long_term_memories = None
        self.short_term_memories = None
        self.short_term_memories_list = []

        # Tracking variables
        self.frame_step = 0
        self.last_long_step = 0
        self.last_short_step = 0

        # Configuration parameters
        self.long_term_mem_gap = cfg.LONG_TERM_MEM_GAP
        self.short_term_mem_skip = cfg.SHORT_TERM_MEM_SKIP
        self.ref_frame_num = cfg.REF_FRAME_NUM
        self.training = False  # Default, can be set via set_training_mode()

    def set_training_mode(self, mode):
        self.training = mode

    def update_memory(self, curr_memories, is_ref=False):
        """Update both short-term and long-term memory based on current memory"""
        # Determine which memories need updating
        update_long = (self.frame_step - self.last_long_step >= self.long_term_mem_gap)
        update_short = (self.cfg.MODEL_FIXED_SHORT_MEM and
                        (self.frame_step - self.last_short_step >= self.short_term_mem_skip)) or \
                       (not self.cfg.MODEL_FIXED_SHORT_MEM)

        if update_long or update_short:
            # Convert memories to 2D format if needed
            memories_2d = self._process_memories_to_2d(curr_memories)

            # Update short-term memory
            if update_short:
                self._update_short_term_memory(memories_2d)

            # Update long-term memory
            if update_long:
                self._update_long_term_memory(curr_memories, is_ref)
                self.last_long_step = self.frame_step

        self.frame_step += 1

    def _process_memories_to_2d(self, memories):
        """Convert memories to 2D format (simplified version)"""
        memories_2d = []
        for layer_mem in memories:
            # Assuming each layer memory is a list/tuple of [k, v] tensors
            layer_2d = [seq_to_2d(m, self.cfg.ENC_SIZE_2D) if m is not None else None
                        for m in layer_mem]
            memories_2d.append(layer_2d)
        return memories_2d

    def _update_short_term_memory(self, new_memories_2d):
        """Update short-term memory storage"""
        if self.cfg.MODEL_FIXED_SHORT_MEM:
            self.last_short_step = self.frame_step
            self.short_term_memories = new_memories_2d
        else:
            self.short_term_memories_list.append(new_memories_2d)
            self.short_term_memories_list = self.short_term_memories_list[-self.short_term_mem_skip:]
            self.short_term_memories = self.short_term_memories_list[0]

    def _update_long_term_memory(self, new_memories, is_ref=False):
        """Update long-term memory storage"""
        if self.long_term_memories is None:
            self.long_term_memories = new_memories
            return

        updated_memories = []
        for new_memory, last_memory in zip(new_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_memory, last_memory):
                if not self.training:
                    if new_e is None or last_e is None:
                        updated_e.append(None)
                    else:
                        e_len = new_e.shape[0]
                        e_num = last_e.shape[0] // e_len
                        max_num = self.cfg.TEST_LONG_TERM_MEM_MAX

                        if max_num <= e_num:
                            last_e = torch.cat([
                                last_e[:e_len * (max_num - (self.ref_frame_num + 1))],
                                last_e[-self.ref_frame_num * e_len:]
                            ], dim=0)

                        updated_e.append(
                            torch.cat([last_e, new_e], dim=0) if is_ref else
                            torch.cat([new_e, last_e], dim=0))
                else:
                    updated_e.append(
                        torch.cat([new_e, last_e], dim=0) if new_e is not None and last_e is not None else None)

            updated_memories.append(updated_e)

        self.long_term_memories = updated_memories


def seq_to_2d(seq_tensor, enc_size_2d):
    """Convert sequence tensor to 2D format (placeholder implementation)"""
    # This should be implemented according to your specific needs
    return seq_tensor  # Placeholder