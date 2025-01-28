"""Test alignment by aligning each example with itself.
"""

import unittest

from codealign import align, Alignment


class TestPerfectSelfAlignment(unittest.TestCase):
    def assert_perfectly_self_aligned(self, code: str, lang: str = "python"):
        alignment: Alignment = align(code, code, lang, injective=True, control_dependence=False)

        candidate_operators = [operator for block in alignment.candidate_ir for operator in block]
        reference_operators = [operator for block in alignment.reference_ir for operator in block]

        self.assertEqual(len(candidate_operators), len(reference_operators))

        for cand, ref in zip(candidate_operators, reference_operators):
            self.assertTrue(alignment[cand] == ref, f"{cand} is self-aligned with {alignment[cand]}.")

    def test_1(self):
        code = """
        def __repr__(self):
            cls_name = self.__class__.__name__
            v_repr = self.__class__._value_repr_ or repr
            if self._name_ is None:
                return '<%s: %s>' % (cls_name, v_repr(self._value_))
            else:
                return '<%s.%s: %s>' % (cls_name, self._name_, v_repr(self._value_))
        """
        self.assert_perfectly_self_aligned(code)

    def test_2(self):
        code = """
        def test(l=200, n=4, fun=sun, startpos=(0, 0), th=2):
            global tiledict
            goto(startpos)
            setheading(0)
            tiledict = {}
            tracer(0)
            fun(l, n)
            draw(l, n, th)
            tracer(1)
            nk = len([x for x in tiledict if tiledict[x]])
            nd = len([x for x in tiledict if not tiledict[x]])
            print('%d kites and %d darts = %d pieces.' % (nk, nd, nk + nd))
        """
        self.assert_perfectly_self_aligned(code)

    def test_3(self):
        code = """
        def _join_parts(self, part_strings):
            return ''.join([part for part in part_strings if part and part is not SUPPRESS])
        """
        self.assert_perfectly_self_aligned(code)

    def test_4(self):
        code = """
        def gauss(self, mu=0.0, sigma=1.0):
            random = self.random
            z = self.gauss_next
            self.gauss_next = None
            if z is None:
                x2pi = random() * TWOPI
                g2rad = _sqrt(-2.0 * _log(1.0 - random()))
                z = _cos(x2pi) * g2rad
                self.gauss_next = _sin(x2pi) * g2rad
            return mu + z * sigma
        """
        self.assert_perfectly_self_aligned(code)

    def test_5(self):
        code = """
        def mn_eck(p, ne, sz):
            turtlelist = [p]
            for i in range(1, ne):
                q = p.clone()
                q.rt(360.0 / ne)
                turtlelist.append(q)
                p = q
            for i in range(ne):
                c = abs(ne / 2.0 - i) / (ne * 0.7)
                for t in turtlelist:
                    t.rt(360.0 / ne)
                    t.pencolor(1 - c, 0, c)
                    t.fd(sz)
        """
        self.assert_perfectly_self_aligned(code)
    
    def test_6(self):
        code = """
        def _splitlines_no_ff(source):
            idx = 0
            lines = []
            next_line = ''
            while idx < len(source):
                c = source[idx]
                next_line += c
                idx += 1
                if c == '\\r' and idx < len(source) and (source[idx] == '\\n'):
                    next_line += '\\n'
                    idx += 1
                if c in '\\r\\n':
                    lines.append(next_line)
                    next_line = ''
            if next_line:
                lines.append(next_line)
            return lines
        """
        self.assert_perfectly_self_aligned(code)

    def test_7(self):
        code = """
        def reduce_pipe_connection(conn):
            access = (_winapi.FILE_GENERIC_READ if conn.readable else 0) | (_winapi.FILE_GENERIC_WRITE if conn.writable else 0)
            dh = reduction.DupHandle(conn.fileno(), access)
            return (rebuild_pipe_connection, (dh, conn.readable, conn.writable))
        """
        self.assert_perfectly_self_aligned(code)
    
    def test_8(self):
        code = """
        def _parse_format_specifier(format_spec, _localeconv=None):
            m = _parse_format_specifier_regex.match(format_spec)
            if m is None:
                raise ValueError('Invalid format specifier: ' + format_spec)
            format_dict = m.groupdict()
            fill = format_dict['fill']
            align = format_dict['align']
            format_dict['zeropad'] = format_dict['zeropad'] is not None
            if format_dict['zeropad']:
                if fill is not None:
                    raise ValueError("Fill character conflicts with '0' in format specifier: " + format_spec)
                if align is not None:
                    raise ValueError("Alignment conflicts with '0' in format specifier: " + format_spec)
            format_dict['fill'] = fill or ' '
            format_dict['align'] = align or '>'
            if format_dict['sign'] is None:
                format_dict['sign'] = '-'
            format_dict['minimumwidth'] = int(format_dict['minimumwidth'] or '0')
            if format_dict['precision'] is not None:
                format_dict['precision'] = int(format_dict['precision'])
            if format_dict['precision'] == 0:
                if format_dict['type'] is None or format_dict['type'] in 'gGn':
                    format_dict['precision'] = 1
            if format_dict['type'] == 'n':
                format_dict['type'] = 'g'
                if _localeconv is None:
                    _localeconv = _locale.localeconv()
                if format_dict['thousands_sep'] is not None:
                    raise ValueError("Explicit thousands separator conflicts with 'n' type in format specifier: " + format_spec)
                format_dict['thousands_sep'] = _localeconv['thousands_sep']
                format_dict['grouping'] = _localeconv['grouping']
                format_dict['decimal_point'] = _localeconv['decimal_point']
            else:
                if format_dict['thousands_sep'] is None:
                    format_dict['thousands_sep'] = ''
                format_dict['grouping'] = [3, 0]
                format_dict['decimal_point'] = '.'
            return format_dict
        """
        self.assert_perfectly_self_aligned(code)
