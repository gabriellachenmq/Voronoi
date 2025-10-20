import numpy as np

pts = np.array([
 [626.50900514, 628.19579636],
 [628.42823511, 627.6406193 ],
 [629.85419649, 627.2927472 ],
 [628.70595672, 628.63193112],
 [627.32133956, 630.06455873],
 [626.79264514, 631.74726669],
 [626.29295353, 633.04928552]
])

crosses = []
n = len(pts)
for i in range(n):
    p0 = pts[i]
    p1 = pts[(i+1)%n]
    p2 = pts[(i+2)%n]
    v1 = p1 - p0
    v2 = p2 - p1
    z = v1[0]*v2[1] - v1[1]*v2[0]
    crosses.append(z)

print("cross z values:", crosses)
print("Any negative while others positive? ->", any(z < 0 for z in crosses) and any(z > 0 for z in crosses))
# find indices of negative crosses (reflex at p_{i+1})
neg_idx = [ (i+1)%n for i,z in enumerate(crosses) if z < 0 ]
print("Reflex vertex indices (0-based):", neg_idx)
for j in neg_idx:
    print("Reflex vertex:", pts[j])
