import os
import sys

sys.path.insert(0, "/Users/macstar/testflight-auto/fastlane-ios-oneclick/scripts")

from asc_api import first_page, make_session  # noqa: E402


session = make_session()
app_id = os.environ.get("ASC_APP_ID") or os.environ.get("APP_STORE_ID") or "6785257642"
build_number = os.environ["APP_BUILD_NUMBER"]
group_name = os.environ.get("TESTFLIGHT_GROUP_NAME") or os.environ.get("BETA_GROUP_NAME") or "PXJ Internal"

builds = first_page(
    session,
    "/builds",
    {
        "filter[app]": app_id,
        "filter[version]": build_number,
        "fields[builds]": "version,processingState,usesNonExemptEncryption,uploadedDate",
        "limit": "5",
    },
)
if not builds:
    raise SystemExit("BUILD_MISSING")
build = builds[0]
print(
    "BUILD "
    f"id={build['id']} "
    f"number={build['attributes'].get('version')} "
    f"state={build['attributes'].get('processingState')} "
    f"encryption={build['attributes'].get('usesNonExemptEncryption')}"
)

groups = first_page(
    session,
    "/betaGroups",
    {
        "filter[app]": app_id,
        "fields[betaGroups]": "name,isInternalGroup,hasAccessToAllBuilds",
        "limit": "100",
    },
)
group = next((item for item in groups if item["attributes"].get("name") == group_name), None)
if not group:
    raise SystemExit("GROUP_MISSING")
print(
    "GROUP "
    f"id={group['id']} "
    f"name={group['attributes'].get('name')} "
    f"internal={group['attributes'].get('isInternalGroup')} "
    f"allBuilds={group['attributes'].get('hasAccessToAllBuilds')}"
)

try:
    build_groups = first_page(
        session,
        f"/builds/{build['id']}/betaGroups",
        {"fields[betaGroups]": "name,isInternalGroup,hasAccessToAllBuilds", "limit": "100"},
    )
    print("BUILD_GROUPS=" + ",".join(item["attributes"].get("name", "") for item in build_groups))
except Exception as exc:
    print("BUILD_GROUPS_QUERY_FAILED=" + exc.__class__.__name__)

try:
    group_builds = first_page(
        session,
        f"/betaGroups/{group['id']}/builds",
        {"fields[builds]": "version,processingState", "limit": "10"},
    )
    print(
        "GROUP_BUILDS="
        + ",".join(
            f"{item['attributes'].get('version')}:{item['attributes'].get('processingState')}"
            for item in group_builds
        )
    )
except Exception as exc:
    print("GROUP_BUILDS_QUERY_FAILED=" + exc.__class__.__name__)
