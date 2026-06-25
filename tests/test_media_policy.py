import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import main


class CardMediaPolicyTests(unittest.TestCase):
    def test_unsafe_sources_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            media_dir = Path(tmp)
            (media_dir / "official.jpg").write_bytes(b"official-image")
            (media_dir / "yelp.jpg").write_bytes(b"yelp-image")
            (media_dir / "openverse.jpg").write_bytes(b"openverse-image")

            rec = {
                "media_items": [
                    {
                        "type": "image",
                        "source": "yelp",
                        "file": "yelp.jpg",
                        "attribution": "Photo: Yelp",
                    },
                    {
                        "type": "image",
                        "source": "openverse",
                        "file": "openverse.jpg",
                        "attribution": "Photo: Openverse",
                    },
                    {
                        "type": "image",
                        "source": "official",
                        "file": "official.jpg",
                        "attribution": "Photo: Official website",
                    },
                ],
            }

            with patch.object(main, "CARD_IMAGES_DIR", str(media_dir)):
                media = main._card_media_from_store(rec)

        self.assertEqual(len(media["media_items"]), 1)
        self.assertEqual(media["media_items"][0].source, "official")
        self.assertEqual(media["image_url"], "/cards/official.jpg")

    def test_duplicate_local_files_are_collapsed_by_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            media_dir = Path(tmp)
            (media_dir / "one.jpg").write_bytes(b"same-image-bytes")
            (media_dir / "two.jpg").write_bytes(b"same-image-bytes")

            rec = {
                "media_items": [
                    {"type": "image", "source": "chicagodoes", "file": "one.jpg"},
                    {"type": "image", "source": "chicagodoes", "file": "two.jpg"},
                ],
            }

            with patch.object(main, "CARD_IMAGES_DIR", str(media_dir)):
                media = main._card_media_from_store(rec)

        self.assertEqual(len(media["media_items"]), 1)
        self.assertEqual(media["media_items"][0].url, "/cards/one.jpg")

    def test_reencoded_duplicate_images_are_collapsed_by_visual_hash(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            media_dir = Path(tmp)
            img = Image.new("RGB", (320, 240), (18, 92, 140))
            img.save(media_dir / "one.jpg")
            img.save(media_dir / "two.webp")

            rec = {
                "media_items": [
                    {"type": "image", "source": "chicagodoes", "file": "one.jpg"},
                    {"type": "image", "source": "chicagodoes", "file": "two.webp"},
                ],
            }

            with patch.object(main, "CARD_IMAGES_DIR", str(media_dir)):
                media = main._card_media_from_store(rec)

        self.assertEqual(len(media["media_items"]), 1)
        self.assertEqual(media["media_items"][0].url, "/cards/one.jpg")

    def test_safe_remote_video_is_kept(self):
        rec = {
            "media_items": [
                {
                    "type": "video",
                    "source": "youtube",
                    "url": "https://www.youtube.com/watch?v=abcdefghijk",
                }
            ],
        }

        media = main._card_media_from_store(rec)

        self.assertEqual(len(media["media_items"]), 1)
        self.assertEqual(media["video_url"], "https://www.youtube.com/watch?v=abcdefghijk")


if __name__ == "__main__":
    unittest.main()
