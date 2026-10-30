"""Click the four court corners in clockwise order, starting at the far-left corner."""

from __future__ import annotations

import argparse
import json

import cv2

from hud.fusion.homography import save_homography


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Still image of the court")
    parser.add_argument("--output", default="config/court_calib.json")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    points: list[list[int]] = []

    def click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(image, (x, y), 6, (0, 255, 255), -1)
            cv2.imshow("Court calibration", image)

    cv2.namedWindow("Court calibration")
    cv2.setMouseCallback("Court calibration", click)
    cv2.imshow("Court calibration", image)
    while len(points) < 4:
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    if len(points) != 4:
        raise SystemExit("Calibration cancelled; exactly four corners are required")
    homography = save_homography(args.output, points)
    print(json.dumps({"corners_px": points, "homography": homography, "output": args.output}))


if __name__ == "__main__":
    main()
