package hf

import "os"

// replaceFile renames tmp onto path, replacing an existing path on Windows.
func replaceFile(tmp, path string) error {
	if err := os.Rename(tmp, path); err == nil {
		return nil
	} else if _, statErr := os.Stat(path); statErr != nil {
		return err
	}

	if err := os.Remove(path); err != nil {
		return err
	}

	return os.Rename(tmp, path)
}
