import torch
from src.data.sampler import LengthGroupedSampler, collate_fn_dynamic_pad


def test_sampler_groups_by_length():
    lengths = [100, 2000, 150, 1800, 120, 2048, 80, 1900]
    batch_size = 2
    
    sampler = LengthGroupedSampler(lengths, batch_size, shuffle=False)
    indices = list(sampler)
    
    # Get lengths for sampled indices
    sampled_lengths = [lengths[i] for i in indices]
    
    # Check that consecutive pairs have similar lengths (sorted)
    for i in range(0, len(sampled_lengths) - 1, batch_size):
        batch_lengths = sampled_lengths[i:i+batch_size]
        assert batch_lengths == sorted(batch_lengths), "Batch should be sorted by length"


def test_sampler_covers_all_indices():
    lengths = [100, 200, 300, 400, 500]
    batch_size = 2
    
    sampler = LengthGroupedSampler(lengths, batch_size, shuffle=False)
    indices = list(sampler)
    
    assert len(indices) == len(lengths)
    assert set(indices) == set(range(len(lengths)))


def test_sampler_shuffle_changes_batch_order():
    lengths = [100, 110, 200, 210, 300, 310]
    batch_size = 2
    
    sampler1 = LengthGroupedSampler(lengths, batch_size, shuffle=True)
    sampler2 = LengthGroupedSampler(lengths, batch_size, shuffle=True)
    
    indices1 = list(sampler1)
    indices2 = list(sampler2)
    
    # Different runs should give different order (probabilistic, but very likely)
    # At least check they're not identical
    assert indices1 != indices2 or len(indices1) < 4  # Small datasets might match

def test_collate_comprehensive():
    
    # Test 5 different batches separately
    test_batches = [
        # Batch 1: Very short sequences (50-100 range) - should pad to 101
        [
            {'input_ids': list(range(1, 51)), 'length': 50},
            {'input_ids': list(range(51, 131)), 'length': 80},
            {'input_ids': list(range(131, 221)), 'length': 90},
            {'input_ids': list(range(221, 322)), 'length': 101},
        ],
        
        # Batch 2: Short sequences (200-300 range) - should pad to 300
        [
            {'input_ids': list(range(322, 522)), 'length': 200},
            {'input_ids': list(range(522, 772)), 'length': 250},
            {'input_ids': list(range(772, 1042)), 'length': 270},
            {'input_ids': list(range(1042, 1342)), 'length': 300},
        ],
        
        # Batch 3: Medium sequences (500-700 range) - should pad to 700
        [
            {'input_ids': list(range(1342, 1842)), 'length': 500},
            {'input_ids': list(range(1842, 2442)), 'length': 600},
            {'input_ids': list(range(2442, 3092)), 'length': 650},
            {'input_ids': list(range(3092, 3792)), 'length': 700},
        ],
        
        # Batch 4: Long sequences (1000-1500 range) - should pad to 1500
        [
            {'input_ids': list(range(3792, 4792)), 'length': 1000},
            {'input_ids': list(range(4792, 6092)), 'length': 1300},
            {'input_ids': list(range(6092, 7492)), 'length': 1400},
            {'input_ids': list(range(7492, 8992)), 'length': 1500},
        ],
        
        # Batch 5: Very long sequences (1800-2048 range) - should pad to 2048
        [
            {'input_ids': list(range(8992, 10792)), 'length': 1800},
            {'input_ids': list(range(10792, 12692)), 'length': 1900},
            {'input_ids': list(range(12692, 14692)), 'length': 2000},
            {'input_ids': list(range(14692, 16740)), 'length': 2048},
        ],
    ]
    
    expected_max_lens = [101, 300, 700, 1500, 2048]
    
    for batch_idx, (batch, expected_max) in enumerate(zip(test_batches, expected_max_lens)):
        result = collate_fn_dynamic_pad(batch, pad_token_id=0)
        
        # Check shape - should pad to max in THIS batch only
        assert result['input_ids'].shape[0] == 4, f"Batch {batch_idx}: Should have 4 samples"
        assert result['input_ids'].shape[1] == expected_max, \
            f"Batch {batch_idx}: Should pad to {expected_max}, got {result['input_ids'].shape[1]}"
        assert result['attention_mask'].shape == result['input_ids'].shape
        
        # Dtype checks
        assert result['input_ids'].dtype == torch.int16
        assert result['attention_mask'].dtype == torch.bool
        
        # Verify each sample in batch
        for i, sample in enumerate(batch):
            expected_length = sample['length']
            expected_ids = sample['input_ids']
            
            # Check actual content
            actual_ids = result['input_ids'][i][:expected_length].tolist()
            assert actual_ids == expected_ids, f"Batch {batch_idx}, Sample {i}: Content mismatch"
            
            # Check padding
            padding = result['input_ids'][i][expected_length:].tolist()
            assert all(p == 0 for p in padding), f"Batch {batch_idx}, Sample {i}: Padding should be all zeros"
            
            # Check attention mask
            mask = result['attention_mask'][i]
            assert mask[:expected_length].all(), f"Batch {batch_idx}, Sample {i}: Real tokens should have mask=True"
            assert not mask[expected_length:].any(), f"Batch {batch_idx}, Sample {i}: Padding should have mask=False"
            
            # Verify mask length matches actual length
            assert mask.sum().item() == expected_length, f"Batch {batch_idx}, Sample {i}: Mask sum should equal length"
        
        # Calculate padding efficiency for this batch
        lengths = [s['length'] for s in batch]
        total_tokens = sum(lengths)
        total_capacity = len(batch) * expected_max
        padding_tokens = total_capacity - total_tokens
        padding_ratio = padding_tokens / total_capacity
        
        print(f"\nBatch {batch_idx + 1} (max_len={expected_max}):")
        print(f"  Lengths: {lengths}")
        print(f"  Total real tokens: {total_tokens:,}")
        print(f"  Total capacity: {total_capacity:,}")
        print(f"  Padding tokens: {padding_tokens:,}")
        print(f"  Padding ratio: {padding_ratio:.2%}")


def test_collate_custom_pad_token():
    batch = [
        {'input_ids': [1, 2], 'length': 2},
        {'input_ids': [3, 4, 5], 'length': 3},
    ]
    
    result = collate_fn_dynamic_pad(batch, pad_token_id=999)
    
    assert result['input_ids'][0].tolist() == [1, 2, 999]
    assert result['input_ids'][1].tolist() == [3, 4, 5]


def test_collate_no_padding_needed():
    batch = [
        {'input_ids': [1, 2, 3], 'length': 3},
        {'input_ids': [4, 5, 6], 'length': 3},
    ]
    
    result = collate_fn_dynamic_pad(batch)
    
    assert result['input_ids'].shape == (2, 3)
    assert result['attention_mask'].all()  # All True (no padding)


def test_sampler_with_single_batch():
    lengths = [100, 110, 120]
    batch_size = 10  # Larger than dataset
    
    sampler = LengthGroupedSampler(lengths, batch_size, shuffle=False)
    indices = list(sampler)
    
    assert len(indices) == 3
    assert set(indices) == {0, 1, 2}


