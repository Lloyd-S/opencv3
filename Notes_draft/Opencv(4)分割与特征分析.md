# Opencv(4)分割

1.图像修补

```python
import numpy as np
import cv2

img = cv2.imread('homework.png')
mask = cv2.imread('mask.png',0)

dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
cv2.imshow('INPAINT_TELEA', dst)
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
cv2.imshow('INPAINT_NS', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

