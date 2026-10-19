"""List the Muse headbands nearby with their names and addresses.

    python find_muse.py

The name is like MuseS-0D57 (the last four characters are also printed on the headband or its
box). On a Mac the address is a long UUID (macOS hides real Bluetooth addresses); on Windows and
Linux it is a MAC address. Pass either to two_states.py: --name MuseS-0D57 or --address <address>.

Nothing listed at all: Bluetooth is off, or the terminal app has no Bluetooth permission
(macOS: System Settings > Privacy & Security > Bluetooth, allow Terminal / iTerm).
Other devices but no Muse: the headband is asleep, off, or already connected elsewhere
(the Muse phone app, another computer). A headband that is free to connect blinks;
a steady light means it is already connected to something.
"""
import asyncio
import sys
from bleak import BleakScanner


async def scan(seconds=12.0):
    return await BleakScanner.discover(timeout=seconds)


def address_of(name):
    for d in asyncio.run(scan()):
        if d.name and d.name.lower() == name.lower():
            return d.address
    raise SystemExit("no headband named %s found nearby (run find_muse.py to see what is around)" % name)


if __name__ == "__main__":
    print("scanning for 12 seconds...")
    devices = asyncio.run(scan())
    muse = [d for d in devices if d.name and "muse" in d.name.lower()]
    if not devices:
        print("nothing at all: Bluetooth is off or this terminal has no Bluetooth permission")
        sys.exit(1)
    if not muse:
        print("%d devices, but no Muse: is the headband on, blinking, and not connected to the phone app?" % len(devices))
        sys.exit(1)
    for d in sorted(muse, key=lambda d: d.name):
        print("   %-16s %s" % (d.name, d.address))
