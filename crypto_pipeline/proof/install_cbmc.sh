#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Donovan Worrell
#
# SPDX-License-Identifier: Apache-2.0
# Catalyst-File-ID: urn:uuid:5e6ec43e-daba-4f62-8098-fd5e452a505c

set -euo pipefail
# Extract the pinned upstream Ubuntu x86-64 package; no root installation.
destination=${1:?Usage: install_cbmc.sh DESTINATION}
mkdir -p "$destination"
package="$destination/cbmc.deb"
curl --fail --location --retry 3 --output "$package" \
  https://github.com/diffblue/cbmc/releases/download/cbmc-6.8.0/ubuntu-22.04-cbmc-6.8.0-Linux.deb
printf '%s  %s\n' 6d8d9736764052d55198be0a96d2d888fb5686c0f4d97b9aa242f50850d6d1f5 "$package" | sha256sum --check
dpkg-deb --extract "$package" "$destination"
"$destination/usr/bin/cbmc" --version
