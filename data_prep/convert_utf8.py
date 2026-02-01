from pathlib import Path
from typing import Iterable

ENCODINGS: Iterable[str] = ("utf-8", "gb18030", "gbk", "big5", "utf-16", "utf-16le", "utf-16be")


def read_with_fallback(path: Path) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc), enc
        except UnicodeError:
            continue
    raise UnicodeError(f"Unable to decode {path} with encodings: {ENCODINGS}")


def convert_to_utf8(path: Path, keep_backup: bool = True) -> str:
    text, used_enc = read_with_fallback(path)
    if keep_backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_bytes(path.read_bytes())
    path.write_text(text, encoding="utf-8")
    return used_enc


def main():
    target = Path("data") / "地煞七十二变.txt"
    if not target.exists():
        raise FileNotFoundError(target)
    used = convert_to_utf8(target, keep_backup=True)
    print(f"Converted {target} from {used} to utf-8. Backup saved next to file.")


if __name__ == "__main__":
    main()
