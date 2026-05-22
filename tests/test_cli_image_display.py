from __future__ import annotations

import hashlib
from io import BytesIO

from app.engine.cli_image_display import (
    CliImageDisplayOptions,
    CliImageDisplayRenderer,
    Iterm2InlineImageBackend,
    PreparedCliImageReveal,
    write_cli_safe_asset_cache,
)
from app.engine.content_asset_bytes import ResolvedAssetBytes
from app.engine.content_assets import write_asset_catalog
from app.schemas.checkpoint import CheckpointFile
from app.schemas.content import ContentPackState
from app.schemas.content_pack import ContentImageAsset, SafeAssetRevealPayload
from app.schemas.responses import TurnResponse
from app.schemas.state import SessionState, WorldState


PACK_ID = "synthetic"
MEDIA_BYTES = b"synthetic reviewed player-safe image bytes"


def test_iterm_backend_supports_only_known_inline_terminals():
    assert Iterm2InlineImageBackend(
        output=BytesIO(),
        environ={"TERM": "xterm-256color", "TERM_PROGRAM": "iTerm.app"},
        is_tty=True,
    ).is_supported()
    assert Iterm2InlineImageBackend(
        output=BytesIO(),
        environ={"TERM": "xterm-256color", "WEZTERM_PANE": "1"},
        is_tty=True,
    ).is_supported()
    assert not Iterm2InlineImageBackend(
        output=BytesIO(),
        environ={"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"},
        is_tty=True,
    ).is_supported()
    assert not Iterm2InlineImageBackend(
        output=BytesIO(),
        environ={"TERM": "dumb", "TERM_PROGRAM": "iTerm.app"},
        is_tty=True,
    ).is_supported()
    assert not Iterm2InlineImageBackend(
        output=BytesIO(),
        environ={
            "TERM": "screen-256color",
            "TERM_PROGRAM": "iTerm.app",
            "TMUX": "/tmp/tmux",
        },
        is_tty=True,
    ).is_supported()


def test_iterm_backend_emits_protocol_without_source_paths_or_asset_refs(tmp_path):
    output = BytesIO()
    backend = Iterm2InlineImageBackend(
        output=output,
        environ={"TERM": "xterm-256color", "TERM_PROGRAM": "iTerm.app"},
        is_tty=True,
    )
    image = PreparedCliImageReveal(
        pov_character_id="aldric",
        cache_path=tmp_path / "session" / "hash.png",
        filename="safe-generated-name.png",
        mime_type="image/png",
        data=MEDIA_BYTES,
        sha256=_sha256(MEDIA_BYTES),
        byte_count=len(MEDIA_BYTES),
    )

    backend.render(image)

    rendered = output.getvalue()
    assert rendered.startswith(b"\x1b]1337;File=")
    assert b"inline=1" in rendered
    assert b"asset://synthetic" not in rendered
    assert b"/private/table" not in rendered
    assert b"source-map" not in rendered


def test_safe_cache_copy_uses_hash_filename_and_not_source_labels(tmp_path):
    resolved = ResolvedAssetBytes(
        pack_id=PACK_ID,
        asset_id="secret-room-map",
        delivery_ref=f"asset://{PACK_ID}/secret-room-map",
        filename="asset-safe.png",
        mime_type="image/png",
        data=MEDIA_BYTES,
        sha256=_sha256(MEDIA_BYTES),
        byte_count=len(MEDIA_BYTES),
    )

    path = write_cli_safe_asset_cache(
        resolved,
        session_id="session named after source-map.png",
        cache_root=tmp_path / "cache",
    )

    assert path.read_bytes() == MEDIA_BYTES
    assert path.name == f"{resolved.sha256}.png"
    assert "secret-room-map" not in str(path)
    assert "source-map" not in str(path)
    assert "asset://" not in str(path)


def test_renderer_resolves_reviewed_asset_and_renders_inline_protocol(tmp_path):
    db_path, media_root, asset = _write_pack_fixture(tmp_path, "rogue-map")
    ckpt = _ckpt_with_pack(db_path, media_root)
    payload = _payload(asset)
    response = TurnResponse(
        session_id="cli_test",
        per_player_asset_reveals={"rogue": [payload]},
    )
    output = BytesIO()
    renderer = CliImageDisplayRenderer(
        backend=Iterm2InlineImageBackend(
            output=output,
            environ={"TERM": "xterm-256color", "TERM_PROGRAM": "iTerm.app"},
            is_tty=True,
        ),
        options=CliImageDisplayOptions(cache_root=tmp_path / "runtime_cache"),
    )

    prepared = renderer.prepare_reveals(
        response,
        ckpt=ckpt,
        session_id="cli_test",
        character_ids={"rogue"},
    )
    result = renderer.render_prepared(prepared["rogue"][0])

    assert result.displayed is True
    assert result.degraded is False
    assert output.getvalue().startswith(b"\x1b]1337;File=")
    cache_path = prepared["rogue"][0].cache_path
    assert cache_path.read_bytes() == MEDIA_BYTES
    assert cache_path.name == f"{asset.sha256}.png"


def test_renderer_degrades_explicitly_on_unsupported_terminal(tmp_path):
    db_path, media_root, asset = _write_pack_fixture(tmp_path, "handout")
    ckpt = _ckpt_with_pack(db_path, media_root)
    response = TurnResponse(
        session_id="cli_test",
        per_player_asset_reveals={"cleric": [_payload(asset)]},
    )
    renderer = CliImageDisplayRenderer(
        backend=Iterm2InlineImageBackend(
            output=BytesIO(),
            environ={"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"},
            is_tty=True,
        ),
        options=CliImageDisplayOptions(
            cache_root=tmp_path / "runtime_cache",
            show_export_path=False,
        ),
    )

    prepared = renderer.prepare_reveals(
        response,
        ckpt=ckpt,
        session_id="cli_test",
        character_ids={"cleric"},
    )
    result = renderer.render_prepared(prepared["cleric"][0])

    assert result.displayed is False
    assert result.degraded is True
    assert result.error_code == "unsupported_terminal"
    assert result.export_path is None


def test_renderer_never_uses_legacy_merged_asset_reveals(tmp_path):
    db_path, media_root, asset = _write_pack_fixture(tmp_path, "merged-map")
    ckpt = _ckpt_with_pack(db_path, media_root)
    response = TurnResponse(
        session_id="cli_test",
        asset_reveals=[_payload(asset)],
        per_player_asset_reveals={},
    )
    renderer = CliImageDisplayRenderer(
        backend=Iterm2InlineImageBackend(
            output=BytesIO(),
            environ={"TERM": "xterm-256color", "TERM_PROGRAM": "iTerm.app"},
            is_tty=True,
        ),
        options=CliImageDisplayOptions(cache_root=tmp_path / "runtime_cache"),
    )

    assert renderer.prepare_reveals(
        response,
        ckpt=ckpt,
        session_id="cli_test",
        character_ids={"rogue"},
    ) == {}


def test_resolution_failure_log_omits_pack_and_asset_ids(caplog, tmp_path):
    secret_pack_id = "privatepack"
    secret_asset_id = "secretroom"
    ckpt = CheckpointFile(
        session=SessionState(session_id="cli_test"),
        world_state=WorldState(),
        characters=[],
    )
    response = TurnResponse(
        session_id="cli_test",
        per_player_asset_reveals={
            "rogue": [
                SafeAssetRevealPayload(
                    pack_id=secret_pack_id,
                    asset_id=secret_asset_id,
                    kind="player_safe_map",
                    title="Safe title",
                    mime_type="image/png",
                    width=64,
                    height=64,
                    sha256="a" * 64,
                    delivery_ref=f"asset://{secret_pack_id}/{secret_asset_id}",
                    presentation="map_overlay",
                    caption="Safe caption.",
                    alt_text="Safe alt text.",
                )
            ],
        },
    )
    renderer = CliImageDisplayRenderer(
        backend=Iterm2InlineImageBackend(
            output=BytesIO(),
            environ={"TERM": "xterm-256color", "TERM_PROGRAM": "iTerm.app"},
            is_tty=True,
        ),
        options=CliImageDisplayOptions(cache_root=tmp_path / "runtime_cache"),
    )

    with caplog.at_level("WARNING", logger="app.engine.cli_image_display"):
        prepared = renderer.prepare_reveals(
            response,
            ckpt=ckpt,
            session_id="cli_test",
            character_ids={"rogue"},
        )

    assert prepared["rogue"][0].error_code == "resolution_failed"
    assert "missing_pack_catalog" in caplog.text
    assert secret_pack_id not in caplog.text
    assert secret_asset_id not in caplog.text


def _write_pack_fixture(
    tmp_path,
    asset_id: str,
) -> tuple[object, object, ContentImageAsset]:
    asset = _asset(asset_id)
    pack_root = tmp_path / "pack"
    media_root = pack_root / "media"
    media_root.mkdir(parents=True)
    (media_root / f"{asset.sha256}.png").write_bytes(MEDIA_BYTES)
    db_path = pack_root / "content.sqlite"
    write_asset_catalog(db_path, [asset])
    return db_path, media_root, asset


def _ckpt_with_pack(db_path, media_root) -> CheckpointFile:
    ckpt = CheckpointFile(
        session=SessionState(session_id="cli_test"),
        world_state=WorldState(),
        characters=[],
    )
    ckpt.session.content_state = {
        PACK_ID: ContentPackState(
            pack_id=PACK_ID,
            metadata={
                "db_path": str(db_path),
                "asset_media_root": str(media_root),
            },
        )
    }
    return ckpt


def _asset(asset_id: str) -> ContentImageAsset:
    return ContentImageAsset(
        pack_id=PACK_ID,
        asset_id=asset_id,
        kind="player_safe_map",
        title="Private title source-map.png",
        mime_type="image/png",
        width=640,
        height=480,
        sha256=_sha256(MEDIA_BYTES),
        source_ref="private-source-ref",
        review_status="approved",
        spoiler_class="low",
        player_safe_alt_text="A safe visible crop.",
        player_safe_caption="A safe visible crop.",
        delivery_ref=f"asset://{PACK_ID}/{asset_id}",
        safe_for_players=True,
        safe_for_llm=False,
        metadata={"safe_label": "fixture"},
    )


def _payload(asset: ContentImageAsset) -> SafeAssetRevealPayload:
    return SafeAssetRevealPayload(
        pack_id=asset.pack_id,
        asset_id=asset.asset_id,
        kind=asset.kind,
        title=asset.title,
        mime_type=asset.mime_type,
        width=asset.width,
        height=asset.height,
        sha256=asset.sha256,
        delivery_ref=asset.delivery_ref,
        presentation="map_overlay",
        caption=asset.player_safe_caption,
        alt_text=asset.player_safe_alt_text,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
