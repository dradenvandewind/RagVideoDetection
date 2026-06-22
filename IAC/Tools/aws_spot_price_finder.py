#!/usr/bin/env python3
"""
Async EC2 Spot Price Finder
Finds the lowest spot price for a given instance type, across all AWS regions.

Usage:
    python spot_price_finder.py [INSTANCE_TYPE]
    python spot_price_finder.py g5g.2xlarge
    python spot_price_finder.py --top 5 c5.xlarge
"""

import asyncio
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass

import aioboto3
import boto3
from botocore.exceptions import ClientError, EndpointResolutionError


@dataclass
class SpotPrice:
    region: str
    az: str
    price: float

    def __str__(self):
        return f"{self.region:<20} {self.az:<30} ${self.price:.6f}"


async def fetch_regions() -> list[str]:
    """Retrieves the list of all available AWS regions."""
    client = boto3.client("ec2", region_name="us-east-1")
    resp = client.describe_regions(AllRegions=False)
    return [r["RegionName"] for r in resp["Regions"]]


async def fetch_spot_prices(
    session: aioboto3.Session,
    region: str,
    instance_type: str,
    now: str,
) -> list[SpotPrice]:
    """Queries the spot price history for a given region."""
    results = []
    try:
        async with session.client("ec2", region_name=region) as ec2:
            paginator = ec2.get_paginator("describe_spot_price_history")
            async for page in paginator.paginate(
                InstanceTypes=[instance_type],
                ProductDescriptions=["Linux/UNIX"],
                StartTime=now,
                PaginationConfig={"MaxItems": 100},
            ):
                for item in page.get("SpotPriceHistory", []):
                    try:
                        price = float(item["SpotPrice"])
                        results.append(
                            SpotPrice(
                                region=region,
                                az=item["AvailabilityZone"],
                                price=price,
                            )
                        )
                    except (ValueError, KeyError):
                        pass
    except (ClientError, EndpointResolutionError, Exception):
        # Region not accessible or instance not available in this region
        pass
    return results


def print_table(prices: list[SpotPrice], top: int | None = None) -> None:
    if not prices:
        print("No prices found.")
        return

    sorted_prices = sorted(prices, key=lambda p: p.price)
    displayed = sorted_prices[:top] if top else sorted_prices

    header = f"{'REGION':<20} {'ZONE':<30} {'PRICE ($/h)'}"
    print(header)
    print("-" * len(header))
    for sp in displayed:
        print(sp)

    print()
    cheapest = sorted_prices[0]
    print("=== CHEAPEST ===")
    print(cheapest)


async def main():
    parser = argparse.ArgumentParser(description="Finds the lowest EC2 spot price.")
    parser.add_argument(
        "instance_type",
        nargs="?",
        default="g5g.2xlarge",
        help="EC2 instance type (default: g5g.2xlarge)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Display only the N cheapest",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of regions queried in parallel (default: 20)",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Instance : {args.instance_type}")
    print(f"Timestamp: {now}")
    print(f"Fetching regions...")

    regions = await fetch_regions()
    print(f"{len(regions)} regions found. Querying in progress...\n")

    session = aioboto3.Session()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded_fetch(region: str) -> list[SpotPrice]:
        async with semaphore:
            return await fetch_spot_prices(session, region, args.instance_type, now)

    tasks = [bounded_fetch(region) for region in regions]
    results = await asyncio.gather(*tasks)

    all_prices: list[SpotPrice] = [sp for sublist in results for sp in sublist]
    print_table(all_prices, top=args.top)


if __name__ == "__main__":
    asyncio.run(main())