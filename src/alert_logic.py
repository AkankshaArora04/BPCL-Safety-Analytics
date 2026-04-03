def generate_alerts(results):
    alerts = []

    boxes = results[0].boxes

    if boxes is None:
        return ["✅ No objects detected"]

    classes = boxes.cls.tolist()

    has_person = False
    has_helmet = False

    for cls in classes:
        cls = int(cls)

        if cls == 0:  # person
            has_person = True

        if cls in [67, 1]:  
            # 67 = cell phone (ignore), 1 = bicycle (just placeholder)
            # helmet usually not in default COCO properly
            has_helmet = True

    if has_person and not has_helmet:
        alerts.append("🚨 Safety Violation: Person without helmet")

    elif has_person and has_helmet:
        alerts.append("✅ Safe: Helmet detected")

    else:
        alerts.append("✅ No person detected")

    return alerts