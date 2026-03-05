import re
import base64
from io import BytesIO
from PIL import Image

def _parse_px(v: str | None) -> int | None:
    """
    把 '300', '300px', '30%' 等解析成像素（百分比直接忽略）
    """
    if not v:
        return None
    v = v.strip().lower()
    if v.endswith("px"):
        v = v[:-2]
    if v.isdigit():
        return int(v)
    return None


def _get_img_size(img):
    """
    尝试从 img 的属性或 style 中解析尺寸
    """
    # 1️⃣ HTML 属性
    w = _parse_px(img.get("width"))
    h = _parse_px(img.get("height"))

    # 2️⃣ style="width:xxx;height:xxx"
    if (w is None or h is None) and img.get("style"):
        style = img["style"]
        mw = re.search(r"width\s*:\s*(\d+)px", style)
        mh = re.search(r"height\s*:\s*(\d+)px", style)
        if w is None and mw:
            w = int(mw.group(1))
        if h is None and mh:
            h = int(mh.group(1))

    return w, h

def _get_size_from_data_uri(data_uri: str) -> tuple[int | None, int | None]:
    """
    从 data:image/...;base64,... 解码真实图片尺寸
    """
    try:
        if not data_uri.startswith("data:image"):
            return None, None

        # data:image/webp;base64,AAAA...
        header, b64 = data_uri.split(",", 1)
        raw = base64.b64decode(b64)

        with Image.open(BytesIO(raw)) as im:
            return im.width, im.height
    except Exception:
        return None, None
    
def _build_css_var_map(raw_html: str) -> dict[str, str]:
    """
    扫描整个 HTML 源码（包含 <style> 内联 CSS），提取：
    --sf-img-16: url(data:image/xxx;base64,....)
    返回 dict: {"--sf-img-16": "data:image/webp;base64,....", ...}
    """
    var_map: dict[str, str] = {}

    # 兼容 url("data:...") / url('data:...') / url(data:...)
    # 兼容有空格/换行/!important
    pattern = re.compile(
        r"(?P<name>--sf-img-[A-Za-z0-9_-]+)\s*:\s*url\(\s*"
        r"(?P<quote>['\"]?)"
        r"(?P<data>data:image[^'\"\)]+)"
        r"(?P=quote)\s*\)",
        re.IGNORECASE | re.DOTALL,
    )

    for m in pattern.finditer(raw_html):
        name = m.group("name")
        data = m.group("data").strip()
        # 有些 SingleFile 会把 data uriing 很长，中间夹换行；DOTALL 已覆盖，但这里再去掉空白更稳
        data = re.sub(r"\s+", "", data)
        var_map[name] = data

    return var_map

def _extract_real_image_data(img, css_var_map: dict[str, str]) -> str | None:
    """
    从 img 本体提取真正 data:image...；支持：
    1) src=data:image...(非svg)
    2) style 里 background-image:url(data:...)
    3) style 里 background-image:var(--sf-img-XX) -> 用 css_var_map 解析
    """
    # 1) src 直接是 data:image（过滤掉占位 svg）
    src = (img.get("src") or "").strip()
    if src.startswith("data:image") and not src.startswith("data:image/svg+xml"):
        return src

    style = img.get("style") or ""

    # 2) inline style 里有 url(data:...)
    m = re.search(r"background-image\s*:\s*url\(\s*(['\"]?)(data:image[^'\"\)]+)\1\s*\)",
                  style, flags=re.IGNORECASE | re.DOTALL)
    if m:
        data = re.sub(r"\s+", "", m.group(2).strip())
        return data

    # 3) inline style 里是 var(--sf-img-XX)
    m = re.search(r"background-image\s*:\s*var\(\s*(--sf-img-[A-Za-z0-9_-]+)\s*\)",
                  style, flags=re.IGNORECASE)
    if m:
        name = m.group(1)
        data = css_var_map.get(name)
        if data:
            return data

    return None
