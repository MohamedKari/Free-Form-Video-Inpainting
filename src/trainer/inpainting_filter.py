from typing import Tuple

class InpaintingFilter:
    def __init__(self, inputs: list, masks: list, window_size: int, noop_mask: int):
        self.window_size = window_size
        self.noop_mask = noop_mask

        self.inputs = inputs
        self.masks = masks
        self.outputs = []

    def get(self) -> Tuple[list, list]:
        if len(self.outputs) == len(self.inputs):
            return None

        if not self.outputs:
            return self.inputs[0:self.window_size], self.masks[0:self.window_size]
        
        assert len(self.outputs) >= self.window_size
        n = len(self.outputs)
        m = self.window_size - 1

        yield_inputs = self.outputs[-m:]
        yield_masks = [self.noop_mask] * m

        yield_inputs.append(self.inputs[n])
        yield_masks.append(self.masks[n])

        return yield_inputs, yield_masks

    def set(self, output_chunk):
        if not self.outputs:
            self.outputs.extend(output_chunk)
        else:
            self.outputs.append(output_chunk[-1])

        return self.inputs, self.masks

def test():

    filter = InpaintingFilter(
        inputs=[0, 1, 2, 3, 4, 5], 
        masks=[100, 101, 102, 103, 104, 105], 
        window_size=3, 
        noop_mask=1)

    input_chunks = [
        [0, 1, 2],
        [101, 204, 3],
        [204, 309, 4],
        [309, 416, 5]
    ]

    mask_chunks = [
        [100, 101, 102],
        [1, 1, 103],
        [1, 1, 104],
        [1, 1, 105]
    ]

    while True:
        next = filter.get()
        if next:
            input_chunk, mask_chunk = next
        else:
            return

        print(input_chunk, mask_chunk)
        expected_input_chunk = input_chunks[0]
        del input_chunks[0]
        
        expected_mask_chunk = mask_chunks[0]
        del mask_chunks[0]

        assert input_chunk == expected_input_chunk, mask_chunk == expected_mask_chunk

        output_chunk = [i * j for i, j in zip(input_chunk, mask_chunk)]
        filter.set(output_chunk)
        print("output_chunk", output_chunk)

test()