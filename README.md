# RouteOptXAI
This repository contains the codes for our ICCAD'23

"Routability Prediction and Optimization Using Explainable AI"
authored by Seonghyeon Park, Daeyeon Kim, Seongbin Kwon, and Seokhyeong Kang


## overall framework
|:--:|
| Overall framework |
<img width="1027" alt="image" src="./framework.png">


 1. extract layout features after early global routing, which are utilized for DRV hotspot prediction.
 2. an XAI technique, DeepSHAP algorithm, compute the contribution of individual features towards the predicted DRV hotspots. 
 3. considering the dominant feature contributions, determine the most suitable optimization method to resolve the predicted DRV hotspots.



## extract feature
<img width="1027" alt="image" src="./feature.png">

- Tiling: we define a grid cell (Gcell) that has a size of seven times the SITE ROW in units of a prediction pixel  
- Feature extraction: we extract a total of 49 features (17 in the placement stage, 32 in the early global route stage) by using ClipGraphExtract
- https://github.com/daeyeon22/ClipGraphExtract.git

### Placement (17 features)

|             |                       |                      |                          |
|:-----------:|:---------------------:|:--------------------:|:------------------------:|
| cell density| number of instances  | RUDY                 | wire density             |
| pin density | number of nets       | local net RUDY       | channel density          |
| flip-flop ratio | number of terminals| global net RUDY      | vertical channel density |
| average terminals | number of global nets | special net RUDY | horizontal channel density |
|             | number of local nets |                      |                          |

### EGR (32 features)

|                 |                     |                          |
|:---------------:|:-------------------:|:------------------------:|
| wire density<sub>i</sub>   | local net density  | vertical channel density |
| channel density<sub>i</sub> | global net density | horizontal channel density |
| via density<sub>i</sub>     | channel density    | worst negative slack     |
|       (where i ∈ {1, · · · , 8} represents the layer number)          |                     | total negative slack     |


## data processing & prediction model
|:--:|
| Overview of data prodessing & prediction model |



