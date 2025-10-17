<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/115161827/ac03dbe0-8f5d-4105-9d21-6e518fc08213"/>

# Anonymize Data
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Results">Results</a> •
  <a href="Acknowledgment">Acknowledgement
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/anonymize-faces)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/anonymize-faces)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/anonymize-faces.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/anonymize-faces.png)](https://supervisely.com)

</div>

## Overview

This application detects faces on images and anonymizes them. It is useful when you need to hide faces on images for privacy reasons.
You can choose between two anonymization methods: solid and blur. You can also choose the shape of the anonymization area: rectangle or ellipse.
A new project will be created with anonymized images. The original images will not be changed.

In some cases you may want to review the results and add/edit annotations. For example, if the app didn't detect some faces, you can add them manually. Or if the app detected a face where there is none, you can delete the annotation. You can also edit the anonymization area if needed. You can do this by following these steps:
1. Run the app with "Save Detections" option enabled.
2. Review the results add/edit annotations if needed.
3. Run the app with "Anonymize" option enabled.

### Release 1.2.0 update

- Renamed app to "Anonymize Data"
- Added "Target selection" option
- Integrated "EgoBlur" neural network model for car licenseplate detection and anonymization.
- App now can target Faces, Car License plates or both.


## How To Run

### Option 1. From ecosystem

0. Find this app in Ecosystem and click `Run Application` button

1. Select Project or Dataset with images you want to anonymize

2. Select the anonymization method: solid or blur

3. Select the shape of the anonymization area: rectangle or ellipse

4. Click `Run` button

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/770000ce-675c-436b-a8ed-2fb34b8ce63d" width="600"/>

### Option 2. From context menu

0. Go to the list of projects in your Workspace or the list of datasets in a project

1. Open the context menu of a project or dataset, select the application in `Run App -> Transform` section

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/b6e96a47-a9b0-4ace-82e9-7bab563d5756"/>

2. Select the anonymization method: solid or blur

3. Select the shape of the anonymization area: rectangle or ellipse

4. Click `Run` button

## Results

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/08481f94-2cfb-4ba4-85d8-10b17cf467d1"/>

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/119248312/55c3f057-1fb7-482a-844b-d742caa09a4e"/>

<img src="https://github.com/supervisely-ecosystem/anonymize-faces/assets/61844772/3f9b0900-206a-43bf-be38-57b80c61c49d"/>

# Acknowledgment

This app is based on the great work `libfacedetection` ([github](https://github.com/ShiqiYu/libfacedetection)). ![GitHub Org's stars](https://img.shields.io/github/stars/ShiqiYu/libfacedetection?style=social) and `EgoBlur` ([github](https://github.com/facebookresearch/EgoBlur)). ![GitHub Org's stars](https://img.shields.io/github/stars/facebookresearch/EgoBlur?style=social)
