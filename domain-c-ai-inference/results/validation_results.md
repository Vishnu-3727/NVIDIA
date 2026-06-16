# Validation Results

## Objective

Evaluate detection accuracy of the optimized UAV obstacle detection model on the VisDrone2019 validation dataset.

---

## Dataset

### VisDrone2019-DET Validation

* Validation Images: 548
* Annotation Files: 548
* Image Resolution: Variable
* Domain: UAV Aerial Object Detection

Classes:

* pedestrian
* people
* bicycle
* car
* van
* truck
* tricycle
* awning-tricycle
* bus
* motor

---

## Overall Metrics

| Metric    | Value |
| --------- | ----- |
| Precision | 0.521 |
| Recall    | 0.389 |
| mAP50     | 0.370 |
| mAP50-95  | 0.221 |

---

## Class-wise Performance

| Class           | mAP50 |
| --------------- | ----- |
| Pedestrian      | 0.17  |
| People          | 0.13  |
| Bicycle         | 0.15  |
| Car             | 0.77  |
| Van             | 0.44  |
| Truck           | 0.29  |
| Tricycle        | 0.27  |
| Awning-Tricycle | 0.31  |
| Bus             | 0.56  |
| Motor           | 0.42  |

---

## Observations

### Strongest Classes

The model achieved the highest performance on:

1. Car (0.77 mAP50)
2. Bus (0.56 mAP50)
3. Van (0.44 mAP50)
4. Motor (0.42 mAP50)

These classes are larger and more visually distinctive in aerial imagery.

### Challenging Classes

Lower performance was observed for:

* Pedestrian
* People
* Bicycle

These classes occupy fewer pixels and exhibit greater appearance variability in UAV imagery.

---

## Validation Summary

The model successfully completed evaluation on the VisDrone validation dataset and maintained acceptable detection quality after deployment optimization.

The strongest detection capability was observed for vehicle-related classes, which are critical for UAV obstacle awareness and navigation applications.

---

## Conclusion

The optimized model achieved:

* Precision: 52.1%
* Recall: 38.9%
* mAP50: 37.0%
* mAP50-95: 22.1%

while delivering real-time inference throughput of up to 120.37 FPS using TensorRT INT8 optimization.
