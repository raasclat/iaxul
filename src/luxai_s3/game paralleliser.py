import asyncio
import os
from subprocess import Popen

async def run_match_async(match_id, players, output_dir):
    output_path = os.path.join(output_dir, f"match_{match_id}.json")
    print(f"Running match {match_id}...")
    process = await asyncio.create_subprocess_exec(
        "python", "Lux-Design-S3\src\luxai_runner\cli.py", 
        players[0], players[1], 
        "--output", output_path,
        "--len", "1000"
    )
    await process.wait()
    print(f"Match {match_id} complete.")

async def main():
    num_matches = 100
    players = ["Lux-Design-S3/kits/python/main.py", "Lux-Design-S3/kits/python2/main.py"]
    output_dir = "matches_results"
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        run_match_async(i, players, output_dir)
        for i in range(num_matches)
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())