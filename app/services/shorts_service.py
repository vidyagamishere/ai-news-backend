import os
import subprocess
import logging
from db_service import PostgreSQLService


async def create_shorts_for_articles(article_ids, yt=True, insta=False):
    db = PostgreSQLService()
    logger = logging.getLogger(__name__)
    processed = []
    logger.info(f"Video creation starts for articles")
    for article_id in article_ids:
        article = db.get_article_by_id(article_id)
        logger.info(f"Video creation starts for article: {article_id}")
        if yt and article["is_yt_shorts"]:
            continue
        if insta and article["is_insta_reels"]:
            continue
        script = f"AI News: {article['title']}. {article['summary']} For details, visit: {article['url']}"
        audio_path = f"./short_{article_id}.mp3"
        video_path = f"./short_{article_id}.mp4"
        try:
            subprocess.run(["gtts-cli", "-l", "en", "-o", audio_path, script], check=True)
            subprocess.run([
                "ffmpeg", "-loop", "1", "-i", article["image_url"], "-i", audio_path,
                "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", "-shortest",
                "-vf", "scale=720:1280", video_path
            ], check=True)
            db.mark_shorts_created(article_id, yt=True,insta=True)
            logger.info(f"Generated audio at {audio_path}")
            logger.info(f"Generated video at {video_path}")
            processed.append({"article_id": article_id, "video_path": video_path, "status": "success"})
        except Exception as e:
            processed.append({"article_id": article_id, "error": str(e), "status": "failed"})
    return processed