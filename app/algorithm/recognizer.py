"""UniversalGraphRecognizer — orchestrator chính cho pipeline nhận diện đồ thị."""

import datetime
import json

import cv2

from .node_detection import detect_nodes
from .ocr import ocr_node_labels
from .weight_detection import detect_weights
from .edge_detection import detect_edges


class UniversalGraphRecognizer:
    """Nhận diện cấu trúc đồ thị từ ảnh.

    Usage:
        recognizer = UniversalGraphRecognizer(image_path)
        recognizer.process()
        data, viz_path = recognizer.export("output_basename")
    """

    def __init__(self, image_path: str):
        """Khởi tạo recognizer với đường dẫn ảnh.

        Args:
            image_path: Đường dẫn tới file ảnh đồ thị.

        Raises:
            ValueError: Nếu không đọc được ảnh.
        """
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        self.img_vis = self.img.copy()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.nodes = []
        self.edges = []
        self.weights = []
        self.logs = []

    def _log(self, msg: str) -> None:
        """Ghi log với timestamp."""
        print(msg)
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{time_str}] {msg}")

    def process(self) -> tuple:
        """Chạy pipeline nhận diện đầy đủ.

        Returns:
            (nodes, edges) tuple.
        """
        self._log("\n" + "=" * 50)
        self._log("BẮT ĐẦU XỬ LÝ")
        self._log("=" * 50 + "\n")

        # Bước 1: Phát hiện nodes
        self._log("🔍 Phát hiện nodes (universal)...")
        self.nodes = detect_nodes(self.img, self.gray)
        self._log(f"Tìm thấy {len(self.nodes)} nodes")

        # Bước 2: OCR nhãn nodes
        self._log("OCR nhãn nodes...")
        ocr_node_labels(self.img, self.img_vis, self.nodes)
        self._log(f"   Labels: {[n['label'] for n in self.nodes]}")

        # Bước 3: Phát hiện trọng số
        self._log("Phát hiện trọng số (Component-based)...")
        self.weights = detect_weights(self.img, self.gray, self.img_vis, self.nodes)
        self._log(f"Tìm thấy {len(self.weights)} trọng số: {[w['value'] for w in self.weights]}")

        # Bước 4: Phát hiện edges + gán trọng số
        self._log("Phát hiện edges...")
        self.edges = detect_edges(self.gray, self.img_vis, self.nodes, self.weights)
        self._log(f"Tìm thấy {len(self.edges)} edges")

        return self.nodes, self.edges

    def output_results(self) -> None:
        """In kết quả ra console."""
        print("\n" + "=" * 50)
        print("KẾT QUẢ")
        print("=" * 50)

        print("\nNODES:")
        for i, node in enumerate(self.nodes):
            print(f"   {i}. {node['label']}")

        print("\nEDGES:")
        if not self.edges:
            print("   (không tìm thấy)")
        else:
            for edge in self.edges:
                from_label = self.nodes[edge['from']]['label']
                to_label = self.nodes[edge['to']]['label']
                weight = edge['weight'] if edge['weight'] else ''

                if weight:
                    print(f"   {from_label} → {to_label} ({weight})")
                else:
                    print(f"   {from_label} → {to_label}")

    def export(self, basename: str = "graph_output") -> tuple:
        """Export kết quả ra JSON và ảnh visualization.

        Args:
            basename: Đường dẫn cơ sở cho output files (không có extension).

        Returns:
            (json_data, viz_path) tuple.
        """
        data = {
            "nodes": [
                {"id": i, "label": n['label']}
                for i, n in enumerate(self.nodes)
            ],
            "edges": [
                {
                    "from": self.nodes[e['from']]['label'],
                    "to": self.nodes[e['to']]['label'],
                    "weight": e['weight']
                }
                for e in self.edges
            ],
            "logs": self.logs
        }

        json_path = f"{basename}.json"
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        viz_path = f"{basename}_visualization.jpg"
        cv2.imwrite(viz_path, self.img_vis)

        print(f"\nĐã lưu: {json_path}, {viz_path}")

        return data, viz_path
