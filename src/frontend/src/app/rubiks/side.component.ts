import { Component, OnInit, Input } from '@angular/core';
import { CommonService } from '../common.service';
import { Side } from './rubiks';

@Component({
  selector: 'app-side',
  styleUrls: ['./side.component.css'],
  template: `
    <div style="height: calc(100%/3); width: 100%;" class="row" *ngFor="let i of [0, 1, 2]">
      <div [class]="getClass(i, j)" *ngFor="let j of [0, 1, 2]"></div>
    </div>
  `,
})
export class SideComponent implements OnInit {

  colours = ["red", "orange", "white", "yellow", "green", "blue"];
  @Input() side: number[][];

  constructor(private commonService: CommonService) { }

  ngOnInit(): void {
  }

  public getClass(i: number, j: number) {
    const colour = this.colours[this.side[i][j]];
    return `sticker col-4 ${colour}`;
  }

}
