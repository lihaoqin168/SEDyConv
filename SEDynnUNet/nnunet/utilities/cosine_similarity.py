import torch
import torch.nn.functional as F

vec1 = torch.FloatTensor([1, 2, 3, 4])
vec2 = torch.FloatTensor([3, 3, 3, 3])

cos_sim = F.cosine_similarity(vec1, vec2, dim=0)
print(cos_sim)